import sys
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from config import (
    BACKTEST_AUDIT_SUMMARY_PATH,
    BASE_PRICE,
    BASE_CAPACITY,
    DATA_END_DATE,
    FORECAST_CHAMPION_PATH,
    FORECAST_OUTPUT_PATH,
    LIVE_COMPETITOR_MARKET_PATH,
    LIVE_MARKET_STATE_PATH,
    MODEL_COMPARISON_PATH,
    OTB_SNAPSHOT_PATH,
)
from pricing_core.engine import calculate_recommended_price
from copilot_core.pricing_agent import run_agentic_pricing
from copilot_core.scenario_copilot import (
    ScenarioChatContext,
    ScenarioConversationMemory,
    update_conversation_memory,
)
from copilot_core.scenario_llm_copilot import handle_grounded_scenario_chat
from pricing_core.local_intel import estimate_local_intel_impact
from copilot_core.manager import (
    build_briefing_payload,
    build_champion_model_audit,
    build_market_outlook_metrics,
    build_opportunity_records,
    build_summary_metrics,
    generate_executive_briefing,
    rank_top_opportunities,
    rank_top_risks,
)
from utils.utility_functions import escape_streamlit_markdown, normalize_reasoning

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

MAX_FORECAST_DATE = DATA_END_DATE + timedelta(days=30)

# 1. Page Configuration
st.set_page_config(page_title="Hotel RMS AI", layout="wide")

# 2. Global title
st.title ("🏨 Hotel Revenue Management System")
st.write("---")

# 3. Helper Functions
def file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(ttl=60, show_spinner=False)
def load_forecast_output_cached(path, mtime):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_model_metrics_cached(path, mtime):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=60, show_spinner=False)
def load_audit_summary_cached(path, mtime):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=60, show_spinner=False)
def load_champion_payload_cached(path, mtime):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_live_market_state_cached(json_path, otb_path, forecast_path, market_path, json_mtime, otb_mtime, forecast_mtime, market_mtime):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            cached_state = json.load(f)
        if cached_state and all("adjusted_otb_occupancy" in entry for entry in cached_state.values()):
            booked_adr_by_date = {}
            if os.path.exists(otb_path):
                otb_adr_df = pd.read_csv(otb_path, usecols=["Date", "OTB_ADR"])
                otb_adr_df["Date"] = pd.to_datetime(otb_adr_df["Date"])
                booked_adr_by_date = {
                    row.Date.strftime("%Y-%m-%d"): float(row.OTB_ADR)
                    for row in otb_adr_df.itertuples(index=False)
                    if pd.notna(row.OTB_ADR)
                }
            for date_key, entry in cached_state.items():
                total_rooms = float(entry.get("total_rooms", BASE_CAPACITY) or BASE_CAPACITY)
                entry.setdefault(
                    "raw_otb_occupancy",
                    float(entry.get("current_otb", 0)) / total_rooms,
                )
                entry.setdefault("stayover_otb", 0)
                entry.setdefault("future_arrival_otb", entry.get("current_otb", 0))
                entry.setdefault("comp_low", entry.get("competitor_price"))
                entry.setdefault("comp_median", entry.get("competitor_price"))
                entry.setdefault("comp_high", entry.get("competitor_price"))
                entry.setdefault("sample_size", 1)
                entry.setdefault("source_quality", "legacy_single_rate")
                entry.setdefault("market_regime", "legacy_single_rate")
                if entry.get("booked_adr") is None:
                    if date_key in booked_adr_by_date:
                        entry["booked_adr"] = booked_adr_by_date[date_key]
                    elif entry.get("otb_adr") is not None:
                        entry["booked_adr"] = entry["otb_adr"]
                legacy_velocity = float(entry.get("booking_velocity", 1.0))
                entry.setdefault("gross_pace_index", legacy_velocity)
                entry.setdefault("retained_pace_index", legacy_velocity)
                entry.setdefault("pickup_trend_index", legacy_velocity)
                entry.setdefault("pricing_pace_index", legacy_velocity)
                entry.setdefault("pace_confidence", "low")
            return cached_state

    if not os.path.exists(otb_path):
        return {}

    otb_df = pd.read_csv(otb_path)
    otb_df["Date"] = pd.to_datetime(otb_df["Date"])

    if os.path.exists(forecast_path):
        forecast_rates = pd.read_csv(forecast_path)
        forecast_rates["Date"] = pd.to_datetime(forecast_rates["Date"])
        if "Competitor_Rate" in forecast_rates.columns:
            otb_df = otb_df.merge(forecast_rates[["Date", "Competitor_Rate"]], on="Date", how="left")
    if os.path.exists(market_path):
        market_df = pd.read_csv(market_path)
        market_df["Date"] = pd.to_datetime(market_df["stay_date"])
        otb_df = otb_df.merge(
            market_df[
                [
                    "Date",
                    "as_of_timestamp",
                    "comp_low",
                    "comp_median",
                    "comp_high",
                    "sample_size",
                    "source_quality",
                    "market_regime",
                ]
            ],
            on="Date",
            how="left",
        )

    state = {}
    for row in otb_df.itertuples(index=False):
        gross_pace_index = float(getattr(row, "Gross_Pace_Index", getattr(row, "Booking_Velocity", 1.0)))
        retained_pace_index = float(getattr(row, "Retained_Pace_Index", gross_pace_index))
        pickup_trend_index = float(getattr(row, "Pickup_Trend_Index", gross_pace_index))
        pricing_pace_index = float(getattr(row, "Pricing_Pace_Index", gross_pace_index))
        status = "Normal"
        if gross_pace_index >= 1.2:
            status = "Ahead of historical pace"
        elif gross_pace_index <= 0.8:
            status = "Behind historical pace"

        competitor_price = getattr(row, "Competitor_Rate", pd.NA)
        if pd.isna(competitor_price):
            competitor_price = getattr(row, "OTB_ADR", pd.NA)
        if pd.isna(competitor_price):
            competitor_price = 120.0
        comp_median = getattr(row, "comp_median", competitor_price)
        if pd.isna(comp_median):
            comp_median = competitor_price
        comp_low = getattr(row, "comp_low", comp_median)
        if pd.isna(comp_low):
            comp_low = comp_median
        comp_high = getattr(row, "comp_high", comp_median)
        if pd.isna(comp_high):
            comp_high = comp_median

        state[pd.to_datetime(row.Date).strftime("%Y-%m-%d")] = {
            "current_otb": int(getattr(row, "Live_OTB", 0)),
            "raw_otb_occupancy": float(getattr(row, "Live_OTB", 0)) / max(getattr(row, "Capacity", BASE_CAPACITY), 1),
            "stayover_otb": int(getattr(row, "Stayover_OTB", 0)),
            "future_arrival_otb": int(getattr(row, "Future_Arrival_OTB", getattr(row, "Live_OTB", 0))),
            "adjusted_otb": float(getattr(row, "Adjusted_OTB", getattr(row, "Live_OTB", 0))),
            "expected_cancellations": float(getattr(row, "Expected_Cancellations", 0.0)),
            "adjusted_otb_occupancy": float(
                getattr(
                    row,
                    "Adjusted_OTB_Occupancy",
                    getattr(row, "Live_OTB", 0) / max(getattr(row, "Capacity", BASE_CAPACITY), 1),
                )
            ),
            "historical_avg_otb": int(getattr(row, "Historical_Avg_OTB", 1)),
            "booked_adr": float(getattr(row, "OTB_ADR", getattr(row, "Competitor_Rate", 0.0))),
            "competitor_price": round(float(comp_median), 2),
            "comp_low": round(float(comp_low), 2),
            "comp_median": round(float(comp_median), 2),
            "comp_high": round(float(comp_high), 2),
            "sample_size": int(getattr(row, "sample_size", 1) if not pd.isna(getattr(row, "sample_size", 1)) else 1),
            "source_quality": getattr(row, "source_quality", "legacy_single_rate"),
            "market_regime": getattr(row, "market_regime", "legacy_single_rate"),
            "market_as_of_timestamp": getattr(row, "as_of_timestamp", None),
            "total_rooms": int(getattr(row, "Capacity", BASE_CAPACITY)),
            "gross_otb": int(getattr(row, "Gross_OTB", getattr(row, "Live_OTB", 0))),
            "net_pickup_7d": int(getattr(row, "Net_Pickup_7d", 0)),
            "historical_net_pickup_7d": int(getattr(row, "Historical_Net_Pickup_7d", 0)),
            "gross_pace_index": gross_pace_index,
            "retained_pace_index": retained_pace_index,
            "pickup_trend_index": pickup_trend_index,
            "pricing_pace_index": pricing_pace_index,
            "pace_confidence": getattr(row, "Pace_Confidence", "low"),
            "booking_velocity": gross_pace_index,
            "status": status,
        }
    return state


def load_live_market_data():
    return load_live_market_state_cached(
        LIVE_MARKET_STATE_PATH,
        OTB_SNAPSHOT_PATH,
        FORECAST_OUTPUT_PATH,
        LIVE_COMPETITOR_MARKET_PATH,
        file_mtime(LIVE_MARKET_STATE_PATH),
        file_mtime(OTB_SNAPSHOT_PATH),
        file_mtime(FORECAST_OUTPUT_PATH),
        file_mtime(LIVE_COMPETITOR_MARKET_PATH),
    )


def load_model_metrics():
    return load_model_metrics_cached(MODEL_COMPARISON_PATH, file_mtime(MODEL_COMPARISON_PATH))


def load_audit_summary():
    return load_audit_summary_cached(BACKTEST_AUDIT_SUMMARY_PATH, file_mtime(BACKTEST_AUDIT_SUMMARY_PATH))


def load_champion_payload():
    return load_champion_payload_cached(FORECAST_CHAMPION_PATH, file_mtime(FORECAST_CHAMPION_PATH))


def adjusted_otb_occupancy(live_entry):
    total_rooms = float(live_entry.get("total_rooms", BASE_CAPACITY) or BASE_CAPACITY)
    if live_entry.get("adjusted_otb_occupancy") is not None:
        return float(live_entry["adjusted_otb_occupancy"])
    if live_entry.get("adjusted_otb") is not None:
        return float(live_entry["adjusted_otb"]) / total_rooms
    return float(live_entry.get("current_otb", 0)) / total_rooms


def humanize_label(value):
    return str(value).replace("_", " ").strip().title()


def compact_metric_display(df):
    if df.empty:
        return df
    display = df.copy()
    if "MAE_pp" not in display.columns and "MAE" in display.columns:
        display["MAE_pp"] = pd.to_numeric(display["MAE"], errors="coerce") * 100
    if "RMSE_pp" not in display.columns and "RMSE" in display.columns:
        display["RMSE_pp"] = pd.to_numeric(display["RMSE"], errors="coerce") * 100
    if "Bias_pp" not in display.columns and "Bias" in display.columns:
        display["Bias_pp"] = pd.to_numeric(display["Bias"], errors="coerce") * 100
    if "Abs_Bias_pp" not in display.columns and "Bias_pp" in display.columns:
        display["Abs_Bias_pp"] = pd.to_numeric(display["Bias_pp"], errors="coerce").abs()

    columns = [
        "Feature_Profile",
        "Model",
        "Strategy",
        "Folds",
        "Observations",
        "MAE_pp",
        "RMSE_pp",
        "Bias_pp",
        "Abs_Bias_pp",
        "MAPE",
        "WAPE",
        "Volatility",
        "Stability",
        "Complexity",
    ]
    labels = {
        "Feature_Profile": "Profile",
        "Observations": "Obs",
        "MAE_pp": "Avg Error % (MAE)",
        "RMSE_pp": "Large Error % (RMSE)",
        "Bias_pp": "Bias %",
        "Abs_Bias_pp": "Abs Bias %",
        "MAPE": "MAPE %",
        "WAPE": "WAPE %",
    }
    display = display[[col for col in columns if col in display.columns]].rename(columns=labels)
    numeric_cols = display.select_dtypes(include="number").columns
    display[numeric_cols] = display[numeric_cols].round(2)
    return display


def format_pct(value):
    return f"{float(value) * 100:.1f}%"


def build_revenue_upside_basis_text(record):
    return (
        f"Upside basis: recommended ADR ${record['recommended_adr']:.2f} × "
        f"{record['expected_rooms']:.2f} expected rooms = ${record['expected_revenue']:,.2f}; "
        f"booked ADR ${record['booked_adr']:.2f} maps to nearest optimizer candidate "
        f"${record['booked_adr_proxy_price']:.2f} × {record['booked_adr_proxy_expected_rooms']:.2f} "
        f"expected rooms = ${record['booked_adr_revenue_proxy']:,.2f}."
    )


def build_today_snapshot_rows(records):
    return [
        {
            "Date": row["date"],
            "ADR": f"${row['recommended_adr']:.2f}",
            "Booked": format_pct(row["raw_otb_occupancy"]),
            "Expected Cancellations": f"{row['expected_cancellations']:.2f}",
            "Likely Retained": format_pct(row["adjusted_otb_occupancy"]),
            "Forecast": format_pct(row["forecasted_occupancy"]),
            "Pickup": f"{row['pickup_trend_index']:.2f}x",
            "Comp Median": f"${row['competitor_median']:.2f}",
            "Upside vs Booked ADR": f"${row['revenue_upside']:,.0f}",
            "Review Status": row["review_status"],
        }
        for row in records
    ]


def build_manager_table_rows(records):
    return [
        {
            "Date": row["date"],
            "ADR": f"${row['recommended_adr']:.2f}",
            "Upside vs Booked ADR": f"${row['revenue_upside']:,.0f}",
            "Booked": format_pct(row["raw_otb_occupancy"]),
            "Forecast": format_pct(row["forecasted_occupancy"]),
            "Why it matters": " ".join(row["top_reasons"]),
        }
        for row in records
    ]


BRIEFING_POLICY_VERSION = 4


@st.cache_data(ttl=300, show_spinner=False)
def executive_briefing_cached(payload_json, policy_version):
    return generate_executive_briefing(json.loads(payload_json))


def render_price_trace(agent_result):
    st.subheader("Price Path")
    price_path_components = agent_result.get("price_path_components", agent_result.get("price_components", []))
    if price_path_components:
        st.dataframe(pd.DataFrame(price_path_components), use_container_width=True, hide_index=True)
    else:
        fallback_rows = [
            {"Driver": "Base rate", "Adjustment": "$+0.00", "Price After": f"${BASE_PRICE:.2f}", "Why": "Starting public rate."},
            {"Driver": "Optimizer ADR", "Adjustment": f"${agent_result['final_adr'] - BASE_PRICE:+.2f}", "Price After": f"${agent_result['final_adr']:.2f}", "Why": "Deterministic candidate-price optimizer output."},
        ]
        st.dataframe(pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True)

    st.subheader("Decision Context")
    decision_context_components = agent_result.get("decision_context_components", [])
    if decision_context_components:
        st.dataframe(pd.DataFrame(decision_context_components), use_container_width=True, hide_index=True)
    else:
        fallback_context_rows = [
            {"Signal": "AI advisory", "Value": "Review only", "Why it matters": "AI reviewed the decision without changing ADR."},
        ]
        st.dataframe(pd.DataFrame(fallback_context_rows), use_container_width=True, hide_index=True)

    guardrail_rows = [{"Guardrail": item} for item in agent_result.get("guardrails_applied", [])]
    if guardrail_rows:
        st.subheader("Guardrail Audit")
        st.dataframe(pd.DataFrame(guardrail_rows), use_container_width=True, hide_index=True)


def render_technical_trace(agent_result, use_expander=True):
    def render_trace_body():
        st.write(f"**Applied Logic Flags:** {agent_result.get('logic_flags', [])}")
        st.write(f"**Forecasted Occupancy:** {agent_result.get('forecasted_occupancy', 0) * 100:.1f}%")
        st.write(f"**Optimizer Diagnostics:** {agent_result.get('optimizer_diagnostics', {})}")
        st.write(f"**Market Context:** {agent_result.get('market_context', {})}")
        st.write(f"**Sold-Out Compression:** {agent_result.get('optimizer_diagnostics', {}).get('sold_out', False)}")
        st.write(f"**AI Recommended Action:** {agent_result.get('ai_recommended_action', 'n/a')}")
        st.write(f"**AI Risk Level:** {agent_result.get('ai_risk_level', 'n/a')}")
        st.write(f"**AI Review Flags:** {agent_result.get('ai_review_flags', [])}")
        st.write(f"**Local Intel Estimate:** {agent_result.get('local_intel_estimate', {})}")
        st.write(f"**Manual Demand Adjustment:** {agent_result.get('manual_demand_shock', 0) * 100:+.1f}%")
        st.write(f"**Local Intel Applied Adjustment:** {agent_result.get('local_intel_applied_shock', 0) * 100:+.1f}%")
        st.write(f"**Total Optimizer Demand Shock:** {agent_result.get('total_demand_shock', 0) * 100:+.1f}%")
        st.write(f"**Guardrails Applied:** {agent_result.get('guardrails_applied', [])}")
        st.write(f"**Manual Approval Required:** {agent_result.get('manual_approval_required', False)}")

    if use_expander:
        with st.expander("Show Technical Trace"):
            render_trace_body()
    else:
        st.subheader("Technical Trace")
        render_trace_body()


def render_scenario_result(agent_result, scenario_state=None, technical_expander=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final ADR", f"${agent_result.get('final_adr', 0.0):.2f}")
    c2.metric("ADR vs Reference", f"{agent_result.get('pct_delta_from_reference', 0):+.2f}%", f"${agent_result.get('absolute_delta', 0):+.2f}")
    c3.metric("Market Gap", f"{agent_result.get('competitor_gap_pct', 0):+.2f}%")
    c4.metric("Pricing Pace", f"{agent_result.get('pricing_pace_index', 1.0)}x")

    st.markdown("---")

    st.subheader("AI Advisory Briefing")
    action_label = agent_result.get("ai_recommended_action", agent_result.get("strategy_applied", "Review Before Publishing"))
    risk_label = agent_result.get("ai_risk_level", "Medium")
    banner_text = f"{action_label} | Risk: {risk_label}"
    if action_label == "Accept Optimizer Price":
        st.success(banner_text)
    elif action_label in ["Hold For Manual Approval", "Investigate Data Quality"]:
        st.error(banner_text)
    else:
        st.warning(banner_text)
    st.info(escape_streamlit_markdown(normalize_reasoning(agent_result.get("strategic_reasoning", ""))))

    result_state = scenario_state or agent_result.get("market_state") or {}
    if result_state:
        scenario_record = {
            "current_otb": float(result_state.get("current_otb", 0.0)),
            "expected_cancellations": float(
                result_state.get("expected_cancellations", agent_result.get("expected_cancellations", 0.0))
            ),
            "adjusted_otb": float(result_state.get("adjusted_otb", result_state.get("current_otb", 0.0))),
        }
        render_selected_booking_quality(scenario_record)

    render_technical_trace(agent_result, use_expander=technical_expander)
    render_price_trace(agent_result)


def append_scenario_chat_response(response, user_message=""):
    answer = response.answer
    sources = []
    for item in list(response.source_labels or []) + list(response.grounding_sources or []):
        if item and item not in sources:
            sources.append(item)
    if response.assumptions:
        answer = f"{answer}\n\nAssumptions: {'; '.join(response.assumptions)}"
    if response.safety_flags:
        answer = f"{answer}\n\nSafety: {'; '.join(response.safety_flags)}"
    if sources:
        answer = f"{answer}\n\nSources: {', '.join(sources)}"
    st.session_state.scenario_copilot_messages.append({"role": "assistant", "content": answer})
    st.session_state.scenario_copilot_pending_draft = (
        response.draft if response.confirmation_prompt else None
    )
    if response.scenario_result:
        st.session_state.scenario_copilot_latest_result = response.scenario_result
    if response.clarification_question:
        st.session_state.scenario_copilot_clarification_count = (
            st.session_state.get("scenario_copilot_clarification_count", 0) + 1
        )
    else:
        st.session_state.scenario_copilot_clarification_count = 0
    st.session_state.scenario_copilot_memory = update_conversation_memory(
        st.session_state.get("scenario_copilot_memory", ScenarioConversationMemory()),
        user_message,
        response,
    )


def reset_scenario_copilot_chat():
    st.session_state.scenario_copilot_messages = [
        {
            "role": "assistant",
            "content": (
                "I can answer Scenario Lab questions, prepare local-intel scenarios, "
                "and run simulations after price-changing inputs are confirmed."
            ),
        }
    ]
    st.session_state.scenario_copilot_pending_draft = None
    st.session_state.scenario_copilot_latest_result = None
    st.session_state.scenario_copilot_clarification_count = 0
    st.session_state.scenario_copilot_memory = ScenarioConversationMemory()


def render_scenario_copilot_chat(context):
    st.subheader("Scenario Copilot Chat")
    st.caption("Ask about the selected date, local intel, market position, pace, or run a confirmed scenario.")

    if "scenario_copilot_messages" not in st.session_state:
        reset_scenario_copilot_chat()
    if "scenario_copilot_pending_draft" not in st.session_state:
        st.session_state.scenario_copilot_pending_draft = None
    if "scenario_copilot_latest_result" not in st.session_state:
        st.session_state.scenario_copilot_latest_result = None
    if "scenario_copilot_clarification_count" not in st.session_state:
        st.session_state.scenario_copilot_clarification_count = 0
    if "scenario_copilot_memory" not in st.session_state:
        st.session_state.scenario_copilot_memory = ScenarioConversationMemory()

    if st.button("Start Over", key="scenario_copilot_start_over"):
        reset_scenario_copilot_chat()
        st.rerun()

    for message in st.session_state.scenario_copilot_messages:
        with st.chat_message(message["role"]):
            st.markdown(escape_streamlit_markdown(message["content"]))

    pending_draft = st.session_state.scenario_copilot_pending_draft
    if pending_draft:
        prompt = "Confirm this scenario before price-changing inputs are applied."
        st.warning(prompt)
        confirm_col, context_col, clear_col = st.columns(3)
        if confirm_col.button("Apply and Run", key="scenario_copilot_apply_run"):
            response = handle_grounded_scenario_chat("confirm and run", context, pending_draft)
            append_scenario_chat_response(response, "confirm and run")
            st.rerun()
        if context_col.button("Run Context Only", key="scenario_copilot_context_only"):
            response = handle_grounded_scenario_chat("run context only", context, pending_draft)
            append_scenario_chat_response(response, "run context only")
            st.rerun()
        if clear_col.button("Clear Draft", key="scenario_copilot_clear"):
            st.session_state.scenario_copilot_pending_draft = None
            st.rerun()

    user_message = st.chat_input("Ask Scenario Copilot")
    if user_message:
        st.session_state.scenario_copilot_messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(escape_streamlit_markdown(user_message))
        response = handle_grounded_scenario_chat(user_message, context, st.session_state.scenario_copilot_pending_draft)
        append_scenario_chat_response(response, user_message)
        if response.confirmation_prompt:
            st.rerun()
        with st.chat_message("assistant"):
            st.markdown(escape_streamlit_markdown(st.session_state.scenario_copilot_messages[-1]["content"]))

    latest_result = st.session_state.get("scenario_copilot_latest_result")
    if latest_result:
        with st.expander("Latest Chat Simulation Result", expanded=True):
            render_scenario_result(latest_result, technical_expander=False)


def build_booking_quality_plot_df(records):
    rows = []
    for row in records:
        total_rooms = max(float(row.get("total_rooms", BASE_CAPACITY) or BASE_CAPACITY), 1.0)
        rows.append(
            {
                "Date": pd.to_datetime(row["date"]),
                "Booked Rooms": float(row.get("current_otb", 0.0)),
                "Stayover Rooms": float(row.get("stayover_otb", 0.0)),
                "Future Arrival Rooms": float(row.get("future_arrival_otb", row.get("current_otb", 0.0))),
                "Likely Retained Rooms": float(row.get("adjusted_otb", 0.0)),
                "Forecast Rooms": min(round(float(row.get("forecasted_occupancy", 0.0)) * total_rooms), total_rooms),
            }
        )
    return pd.DataFrame(rows)


def render_booking_quality_trend(records):
    plot_df = build_booking_quality_plot_df(records)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Booked Rooms"],
            name="Booked Rooms",
            line=dict(color="#2563EB", width=3),
            customdata=plot_df[["Stayover Rooms", "Future Arrival Rooms"]],
            hovertemplate=(
                "Booked Rooms: %{y:.2f}<br>"
                "Stayover Rooms: %{customdata[0]:.2f}<br>"
                "Future Arrival Rooms: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Likely Retained Rooms"],
            name="Likely Retained Rooms",
            line=dict(color="#0F766E", width=3),
            customdata=plot_df[["Stayover Rooms", "Future Arrival Rooms"]],
            hovertemplate=(
                "Likely Retained Rooms: %{y:.2f}<br>"
                "Stayover Rooms: %{customdata[0]:.2f}<br>"
                "Future Arrival Rooms: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Forecast Rooms"],
            name="Forecast Rooms",
            line=dict(color="#F59E0B", width=3, dash="dot"),
            customdata=plot_df[["Stayover Rooms", "Future Arrival Rooms"]],
            hovertemplate=(
                "Forecast Rooms: %{y:.2f}<br>"
                "Stayover Rooms: %{customdata[0]:.2f}<br>"
                "Future Arrival Rooms: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="30-Day Booking Quality",
        hovermode="x unified",
        yaxis_title="Room Nights",
        legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_selected_booking_quality(record):
    values = [
        float(record.get("current_otb", 0.0)),
        float(record.get("expected_cancellations", 0.0)),
        float(record.get("adjusted_otb", 0.0)),
    ]
    fig = go.Figure(
        go.Bar(
            x=["Current Booked", "Expected Cancellations", "Likely Retained"],
            y=values,
            marker_color=["#2563EB", "#DC2626", "#0F766E"],
            text=[f"{value:.2f}" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Selected-Date Booking Quality",
        yaxis_title="Rooms",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cancellation risk is estimated only for not-yet-arrived bookings using remaining days to arrival, lead time, "
        "segment, channel, and customer type; in-house stayovers are treated as retained."
    )


def render_pricing_vs_market_chart(records):
    plot_df = pd.DataFrame(
        {
            "Date": pd.to_datetime([row["date"] for row in records]),
            "Recommended ADR": [row["recommended_adr"] for row in records],
            "Comp Low": [row["comp_low"] for row in records],
            "Comp Median": [row["competitor_median"] for row in records],
            "Comp High": [row["comp_high"] for row in records],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Comp High"],
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Comp Low"],
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.22)",
            line=dict(width=0),
            name="Comp Range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Comp Median"],
            name="Comp Median",
            line=dict(color="#64748B", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Recommended ADR"],
            name="Recommended ADR",
            line=dict(color="#DB2777", width=3),
        )
    )
    fig.update_layout(
        title="Pricing vs Market",
        hovermode="x unified",
        yaxis_title="ADR ($)",
        legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

# 4. Sidebar Navigation
st.sidebar.header("🕹️ System Controls")
app_mode = st.sidebar.radio("Switch View", ["Morning Briefing", "Market Outlook", "Scenario Lab"])
forecast_df = load_forecast_output_cached(FORECAST_OUTPUT_PATH, file_mtime(FORECAST_OUTPUT_PATH))

# ==========================================
# PAGE 1: MORNING BRIEFING
# ==========================================
if app_mode == "Morning Briefing":
    st.subheader("Morning Briefing")
    st.caption(f"Demo as-of date: {DATA_END_DATE.strftime('%Y-%m-%d')}")
    live_data = load_live_market_data()
    opportunity_records = build_opportunity_records(forecast_df, live_data)
    top_opportunities = rank_top_opportunities(opportunity_records)
    top_risks = rank_top_risks(opportunity_records)
    summary_metrics = build_summary_metrics(opportunity_records)
    briefing_payload = build_briefing_payload(opportunity_records)
    briefing = executive_briefing_cached(json.dumps(briefing_payload, sort_keys=True), BRIEFING_POLICY_VERSION)

    st.subheader("Executive Briefing")
    st.info(escape_streamlit_markdown(normalize_reasoning(briefing)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue Upside", f"${summary_metrics['total_revenue_upside']:,.0f}")
    c2.metric("Dates With Upside", summary_metrics["dates_with_upside"])
    c3.metric("Dates Needing Review", summary_metrics["dates_needing_review"])
    c4.metric("Sold-Out Dates", summary_metrics["sold_out_dates"])

    render_booking_quality_trend(opportunity_records)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Revenue Opportunities")
        if top_opportunities:
            st.dataframe(
                pd.DataFrame(build_manager_table_rows(top_opportunities)),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No material revenue upside stands out right now.")
    with c2:
        st.subheader("Top Risks / Review Needed")
        if top_risks:
            risk_rows = [
                {
                    "Date": row["date"],
                    "ADR": f"${row['recommended_adr']:.2f}",
                    "Review Status": row["review_status"],
                    "Why review": " ".join(row["review_flags"][:2]),
                }
                for row in top_risks
            ]
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No dates currently require special review before publishing.")

    st.subheader("30-Day Snapshot")
    st.dataframe(
        pd.DataFrame(build_today_snapshot_rows(opportunity_records)),
        use_container_width=True,
        height=360,
        hide_index=True,
    )

    inspectable_dates = [record["date"] for record in opportunity_records]
    default_date = top_opportunities[0]["date"] if top_opportunities else inspectable_dates[0]
    selected_date = st.selectbox(
        "Inspect date",
        inspectable_dates,
        index=inspectable_dates.index(default_date),
        help="Choose any date from the opportunity or risk tables to inspect the full pricing trace.",
    )
    selected_record = next(record for record in opportunity_records if record["date"] == selected_date)
    selected_state = live_data.get(selected_date, {})
    agent_result = run_agentic_pricing(
        target_date=selected_date,
        current_occupancy=selected_record["adjusted_otb_occupancy"],
        forecasted_occupancy=selected_record["forecasted_occupancy"],
        shock=0.0,
        competitor_price=selected_record["competitor_median"],
        market_context={
            "comp_low": selected_state.get("comp_low"),
            "comp_median": selected_state.get("comp_median"),
            "comp_high": selected_state.get("comp_high"),
            "sample_size": selected_state.get("sample_size", 1),
            "source_quality": selected_state.get("source_quality"),
            "market_regime": selected_state.get("market_regime"),
            "market_as_of_timestamp": selected_state.get("market_as_of_timestamp"),
        },
        booking_velocity=float(selected_state.get("booking_velocity", 1.0)),
        gross_pace_index=float(selected_state.get("gross_pace_index", selected_state.get("booking_velocity", 1.0))),
        retained_pace_index=float(selected_state.get("retained_pace_index", selected_state.get("booking_velocity", 1.0))),
        pickup_trend_index=float(selected_state.get("pickup_trend_index", selected_state.get("booking_velocity", 1.0))),
        pricing_pace_index=float(selected_state.get("pricing_pace_index", selected_state.get("booking_velocity", 1.0))),
        historical_avg_otb=int(selected_state.get("historical_avg_otb", 1)),
        market_state=selected_state,
        raw_otb_occupancy=selected_record["raw_otb_occupancy"],
        adjusted_otb_occupancy=selected_record["adjusted_otb_occupancy"],
        expected_cancellations=float(selected_state.get("expected_cancellations", 0.0)),
        record_decision=False,
    )

    st.subheader(f"Date Detail — {selected_date}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommended ADR", f"${selected_record['recommended_adr']:.2f}")
    c2.metric("Current Booked", format_pct(selected_record["raw_otb_occupancy"]))
    c3.metric("Expected Cancellations", f"{selected_record['expected_cancellations']:.2f}")
    c4.metric("Likely Retained", format_pct(selected_record["adjusted_otb_occupancy"]))
    c5.metric("Forecast Occupancy", format_pct(selected_record["forecasted_occupancy"]))
    st.caption(
        escape_streamlit_markdown(
            f"Competitor median: ${selected_record['competitor_median']:.2f} | "
            f"Booked ADR: ${selected_record['booked_adr']:.2f} | "
            f"Upside vs booked ADR: ${selected_record['revenue_upside']:,.0f} | "
            f"Review status: {selected_record['review_status']} | "
            f"Stayovers: {selected_record.get('stayover_otb', 0):.0f} | "
            f"Future arrivals: {selected_record.get('future_arrival_otb', selected_record['current_otb']):.0f}"
        )
    )
    st.caption(escape_streamlit_markdown(build_revenue_upside_basis_text(selected_record)))
    render_selected_booking_quality(selected_record)
    st.info(escape_streamlit_markdown(normalize_reasoning(agent_result["strategic_reasoning"])))
    render_technical_trace(agent_result)
    render_price_trace(agent_result)

# ==========================================
# PAGE 2: MARKET OUTLOOK
# ==========================================
elif app_mode == "Market Outlook":
    st.subheader("Market Outlook")
    st.caption(f"Demo as-of date: {DATA_END_DATE.strftime('%Y-%m-%d')}")
     
    live_data = load_live_market_data()
    metrics_df = load_model_metrics()
    audit_summary_df = load_audit_summary()
    champion_payload = load_champion_payload()
    opportunity_records = build_opportunity_records(forecast_df, live_data)
    outlook_metrics = build_market_outlook_metrics(opportunity_records)
    champion_audit = build_champion_model_audit(champion_payload, audit_summary_df)

    c1, c2, c3, c4 = st.columns(4)
    recent_avg_miss_pp = champion_audit["recent_avg_occupancy_miss_pp"]
    c1.metric("30-Day Booked Room Nights", f"{outlook_metrics['booked_room_nights']:,.0f}")
    c2.metric("Likely Retained Room Nights", f"{outlook_metrics['retained_room_nights']:,.0f}")
    c3.metric("Recent Avg Occupancy Miss (MAE)", "n/a" if pd.isna(recent_avg_miss_pp) else f"{recent_avg_miss_pp:.1f} pp")
    c4.metric("High-Demand Market Dates", outlook_metrics["high_demand_market_dates"])

    render_booking_quality_trend(opportunity_records)
    render_pricing_vs_market_chart(opportunity_records)

    # Inject Live Data into the Forecast Dataframe for the table
    live_records = []
    for index, row in forecast_df.iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d')
        live_entry = live_data.get(d_str, {})
        
        # Calculate Current Baseline Price using Pricing Engine
        # price, rules = calculate_recommended_price(
        #     occupancy= max(live_entry.get('current_otb', 0)/BASE_CAPACITY,row['Forecasted_Occupancy']),
        #     day_name=row['Date'].strftime('%A'),
        #     target_date = row['Date'],
        #     competitor_price=live_entry.get('competitor_price', 120.0)
        # )

        price, rules = calculate_recommended_price(
            occupancy=max(adjusted_otb_occupancy(live_entry), float(row['Forecasted_Occupancy'])),
            day_name=row['Date'].strftime('%A'),
            target_date=d_str,
            competitor_price=live_entry.get('competitor_price', 120.0),
            booking_velocity=live_entry.get('booking_velocity', 1.0),
            gross_pace_index=live_entry.get('gross_pace_index'),
            retained_pace_index=live_entry.get('retained_pace_index'),
            pickup_trend_index=live_entry.get('pickup_trend_index'),
            pricing_pace_index=live_entry.get('pricing_pace_index'),
            raw_otb_occupancy=float(live_entry.get('current_otb', 0)) / float(live_entry.get('total_rooms', BASE_CAPACITY) or BASE_CAPACITY),
            adjusted_otb_occupancy=adjusted_otb_occupancy(live_entry),
            expected_cancellations=live_entry.get('expected_cancellations', 0.0),
        )
        
        live_records.append({
            "Date": d_str,
            "Day": row['Date'].strftime('%A'),
            "Booked Rooms": f"{live_entry.get('current_otb', 0)}",
            "Booked Occupancy": f"{live_entry.get('current_otb', 0)/BASE_CAPACITY:.1%}",
            "Forecast Rooms": f"{min(round(row['Forecasted_Occupancy']*BASE_CAPACITY),237)}",
            "Comp Median": f"${live_entry.get('competitor_price', 0):.2f}",
            "Recommended ADR": f"${price:.2f}",
            "Booked Pace": f"{live_entry.get('gross_pace_index', live_entry.get('booking_velocity', 1.0)):.2f}x",
            "Recent Pickup": f"{live_entry.get('pickup_trend_index', live_entry.get('booking_velocity', 1.0)):.2f}x",
            "Status": live_entry.get('status', 'Normal')
        })

    # Strategy Table
    st.subheader("30-Day Strategy Feed")
    st.dataframe(pd.DataFrame(live_records), use_container_width=True, height=400, hide_index=True)

    st.subheader("Champion Model Audit")
    st.caption(f"Champion: {humanize_label(champion_audit['champion_model'])}")
    st.dataframe(pd.DataFrame(champion_audit["rows"]), use_container_width=True, hide_index=True)
    with st.expander("Show full model comparison"):
        if not metrics_df.empty:
            st.dataframe(compact_metric_display(metrics_df), use_container_width=True, height=260, hide_index=True)
        else:
            st.info("Run forecast backtesting to populate model comparison metrics.")

# ==========================================
# PAGE 3: SCENARIO LAB
# ==========================================
else:
    st.subheader("Scenario Lab")
    
    # Sidebar Sliders
    st.sidebar.divider()
    target_date = st.sidebar.date_input("Target Date",
    value=DATA_END_DATE + timedelta(days=1), # Default to the first day of forecast
    min_value=DATA_END_DATE + timedelta(days=1),
    max_value=MAX_FORECAST_DATE )
    d_str = target_date.strftime('%Y-%m-%d')
    
    # Fetch CURRENT PMS-derived live state for side-bar context
    live_market = load_live_market_data()
    current_state = live_market.get(
        d_str,
        {
            "current_otb": 50,
            "raw_otb_occupancy": 50 / BASE_CAPACITY,
            "adjusted_otb": 50.0,
            "expected_cancellations": 0.0,
            "adjusted_otb_occupancy": 50 / BASE_CAPACITY,
            "historical_avg_otb": 1,
            "competitor_price": 120.0,
            "gross_pace_index": 1.0,
            "retained_pace_index": 1.0,
            "pickup_trend_index": 1.0,
            "pricing_pace_index": 1.0,
            "booking_velocity": 1.0,
            "status": "Fallback snapshot",
        },
    )

    st.sidebar.info(
        escape_streamlit_markdown(
            f"Current booked rooms: {current_state['current_otb']} | "
            f"Comp set (low / median / high): "
            f"${current_state.get('comp_low', current_state['competitor_price']):.2f}"
            f" / ${current_state.get('comp_median', current_state['competitor_price']):.2f}"
            f" / ${current_state.get('comp_high', current_state['competitor_price']):.2f}"
        )
    )
    st.sidebar.caption(
        f"Market regime: {current_state.get('market_regime', 'n/a').replace('_', ' ').title()} | "
        f"Source: {current_state.get('source_quality', 'n/a')}"
    )
    with st.sidebar.expander("Scenario-only market override"):
        override_market = st.checkbox("Override market feed for this run", value=False)
        override_low = st.number_input("Comp low", value=float(current_state.get("comp_low", current_state["competitor_price"])))
        override_median = st.number_input("Comp median", value=float(current_state.get("comp_median", current_state["competitor_price"])))
        override_high = st.number_input("Comp high", value=float(current_state.get("comp_high", current_state["competitor_price"])))
    st.sidebar.subheader("Local Intel")
    manual_event = st.sidebar.text_input("Enter local event (e.g., '100-person wedding block')")

    target_date_parsed = pd.to_datetime(target_date, format="%Y/%m/%d")
    result = forecast_df.loc[forecast_df["Date"] == target_date_parsed, "Forecasted_Occupancy"]
    forecasted_occ = float(result.iloc[0]) if not result.empty else float(current_state['current_otb'] / BASE_CAPACITY)
    raw_current_occ = float(current_state['current_otb'] / BASE_CAPACITY)
    current_occ = adjusted_otb_occupancy(current_state)

    local_intel_estimate = estimate_local_intel_impact(
        manual_event,
        current_occ=current_occ,
        forecast_occ=forecasted_occ,
        booking_velocity=float(current_state.get("booking_velocity", 1.0)),
        retained_pace_index=float(current_state.get("retained_pace_index", current_state.get("booking_velocity", 1.0))),
        pickup_trend_index=float(current_state.get("pickup_trend_index", current_state.get("booking_velocity", 1.0))),
    )

    if manual_event:
        st.sidebar.info("Local intel estimated as decision support. It is not applied unless you approve it.")
        st.sidebar.caption(
            f"Type: {local_intel_estimate['classification']} | "
            f"Suggested impact: {local_intel_estimate['suggested_shock_pct']:+.1f}% | "
            f"Confidence: {local_intel_estimate['confidence']}"
        )
        st.sidebar.caption(local_intel_estimate["rationale"])
        apply_local_intel = st.sidebar.checkbox(
            "Apply local intel estimate to baseline",
            value=False,
            disabled=not local_intel_estimate["apply_allowed"],
        )
        if not local_intel_estimate["apply_allowed"]:
            st.sidebar.caption("Guardrail: this intel is context-only unless clearer room-demand evidence is supplied.")
    else:
        apply_local_intel = False

    demand_shock = st.sidebar.slider("Manual Demand Adjustment (%)", -30, 30, 0) / 100.0
    local_intel_applied_shock = local_intel_estimate["suggested_shock"] if apply_local_intel else 0.0
    total_baseline_shock = demand_shock + local_intel_applied_shock
    st.sidebar.caption(f"Total demand shock included in optimizer: {total_baseline_shock * 100:+.1f}%")

    scenario_horizon_records = build_opportunity_records(forecast_df, live_market)
    scenario_horizon_summary = build_summary_metrics(scenario_horizon_records)
    scenario_chat_context = ScenarioChatContext(
        target_date=d_str,
        forecasted_occupancy=float(forecasted_occ),
        current_state=current_state,
        manual_demand_shock=demand_shock,
        latest_result=st.session_state.get("scenario_copilot_latest_result"),
        live_market_by_date=live_market,
        forecast_occupancy_by_date={
            row.Date.strftime("%Y-%m-%d"): float(row.Forecasted_Occupancy)
            for row in forecast_df.itertuples(index=False)
        },
        clarification_count=st.session_state.get("scenario_copilot_clarification_count", 0),
        conversation_memory=st.session_state.get("scenario_copilot_memory", ScenarioConversationMemory()),
        horizon_records=scenario_horizon_records,
        horizon_summary=scenario_horizon_summary,
    )
    render_scenario_copilot_chat(scenario_chat_context)

    if st.sidebar.button("Run Scenario", type="primary"):
        market_context = {
            "comp_low": override_low if override_market else current_state.get("comp_low"),
            "comp_median": override_median if override_market else current_state.get("comp_median"),
            "comp_high": override_high if override_market else current_state.get("comp_high"),
            "sample_size": current_state.get("sample_size", 1),
            "source_quality": "manual_override" if override_market else current_state.get("source_quality"),
            "market_regime": "manual_override" if override_market else current_state.get("market_regime"),
            "market_as_of_timestamp": current_state.get("market_as_of_timestamp"),
        }
        # --- THE LIVE LOG ---
        log_placeholder = st.empty()
        with log_placeholder.container():
            st.write("🔍 **Node 1: Data Ingestion** - Using cached PMS-derived market snapshot...")
            st.write("📐 **Node 2: Price Optimizer** - Evaluating candidate ADRs and expected revenue...")
            st.write("📈 **Node 3: Pace Analyst** - Reading booked pace and recent pickup...")
            st.write("🧠 **Node 4: AI Strategist** - Reviewing optimizer rationale and risk flags...")

        agent_result = run_agentic_pricing(
            target_date=d_str,
            current_occupancy=current_occ,
            forecasted_occupancy=float(forecasted_occ),
            shock=demand_shock,
            manual_event_text=manual_event,
            competitor_price=float(market_context.get("comp_median") or current_state.get("competitor_price", 120.0)),
            market_context=market_context,
            booking_velocity=float(current_state.get("booking_velocity", 1.0)),
            gross_pace_index=float(current_state.get("gross_pace_index", current_state.get("booking_velocity", 1.0))),
            retained_pace_index=float(current_state.get("retained_pace_index", current_state.get("booking_velocity", 1.0))),
            pickup_trend_index=float(current_state.get("pickup_trend_index", current_state.get("booking_velocity", 1.0))),
            pricing_pace_index=float(current_state.get("pricing_pace_index", current_state.get("booking_velocity", 1.0))),
            historical_avg_otb=int(current_state.get("historical_avg_otb", 1)),
            market_state=current_state,
            manual_demand_shock=demand_shock,
            local_intel_estimate=local_intel_estimate,
            local_intel_applied_shock=local_intel_applied_shock,
            raw_otb_occupancy=raw_current_occ,
            adjusted_otb_occupancy=current_occ,
            expected_cancellations=float(current_state.get("expected_cancellations", 0.0)),
        )

        # Clear logs and show results
        log_placeholder.empty()
        render_scenario_result(agent_result, scenario_state=current_state)
