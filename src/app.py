import sys
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from plotly.subplots import make_subplots
from config import BASE_PRICE,BASE_CAPACITY, DATA_END_DATE, FORECAST_OUTPUT_PATH, LIVE_COMPETITOR_MARKET_PATH, LIVE_MARKET_STATE_PATH, MODEL_COMPARISON_PATH, OTB_SNAPSHOT_PATH
from pricing_engine import calculate_recommended_price
from pricing_agent import run_agentic_pricing
from local_intel_estimator import estimate_local_intel_impact
from manager_copilot import (
    build_briefing_payload,
    build_opportunity_records,
    generate_executive_briefing,
    rank_top_opportunities,
    rank_top_risks,
)
from utils.utility_functions import normalize_reasoning

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
def load_live_market_state_cached(json_path, otb_path, forecast_path, market_path, json_mtime, otb_mtime, forecast_mtime, market_mtime):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            cached_state = json.load(f)
        if cached_state and all("adjusted_otb_occupancy" in entry for entry in cached_state.values()):
            for entry in cached_state.values():
                total_rooms = float(entry.get("total_rooms", BASE_CAPACITY) or BASE_CAPACITY)
                entry.setdefault(
                    "raw_otb_occupancy",
                    float(entry.get("current_otb", 0)) / total_rooms,
                )
                entry.setdefault("comp_low", entry.get("competitor_price"))
                entry.setdefault("comp_median", entry.get("competitor_price"))
                entry.setdefault("comp_high", entry.get("competitor_price"))
                entry.setdefault("sample_size", 1)
                entry.setdefault("source_quality", "legacy_single_rate")
                entry.setdefault("market_regime", "legacy_single_rate")
                entry.setdefault("booked_adr", entry.get("otb_adr", entry.get("competitor_price")))
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


def adjusted_otb_occupancy(live_entry):
    total_rooms = float(live_entry.get("total_rooms", BASE_CAPACITY) or BASE_CAPACITY)
    if live_entry.get("adjusted_otb_occupancy") is not None:
        return float(live_entry["adjusted_otb_occupancy"])
    if live_entry.get("adjusted_otb") is not None:
        return float(live_entry["adjusted_otb"]) / total_rooms
    return float(live_entry.get("current_otb", 0)) / total_rooms


def humanize_label(value):
    return str(value).replace("_", " ").strip().title()


def format_pct(value):
    return f"{float(value) * 100:.1f}%"


def build_today_snapshot_rows(records):
    return [
        {
            "Date": row["date"],
            "ADR": f"${row['recommended_adr']:.2f}",
            "Booked": format_pct(row["raw_otb_occupancy"]),
            "Retained OTB": format_pct(row["adjusted_otb_occupancy"]),
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


BRIEFING_POLICY_VERSION = 3


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

# 4. Sidebar Navigation
st.sidebar.header("🕹️ System Controls")
app_mode = st.sidebar.radio("Switch View", ["Today", "📈 Market Performance", "🤖 Agentic Simulation"])
forecast_df = load_forecast_output_cached(FORECAST_OUTPUT_PATH, file_mtime(FORECAST_OUTPUT_PATH))

# ==========================================
# PAGE 1: TODAY (Manager Copilot)
# ==========================================
if app_mode == "Today":
    st.subheader("Today")
    live_data = load_live_market_data()
    opportunity_records = build_opportunity_records(forecast_df, live_data)
    top_opportunities = rank_top_opportunities(opportunity_records)
    top_risks = rank_top_risks(opportunity_records)
    briefing_payload = build_briefing_payload(opportunity_records)
    briefing = executive_briefing_cached(json.dumps(briefing_payload, sort_keys=True), BRIEFING_POLICY_VERSION)

    st.subheader("Executive Briefing")
    st.info(normalize_reasoning(briefing))

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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended ADR", f"${selected_record['recommended_adr']:.2f}")
    c2.metric("Current Booked", format_pct(selected_record["raw_otb_occupancy"]))
    c3.metric("Retained OTB", format_pct(selected_record["adjusted_otb_occupancy"]))
    c4.metric("Forecast Occupancy", format_pct(selected_record["forecasted_occupancy"]))
    st.caption(
        f"Competitor median: ${selected_record['competitor_median']:.2f} | "
        f"Booked ADR: ${selected_record['booked_adr']:.2f} | "
        f"Upside vs booked ADR: ${selected_record['revenue_upside']:,.0f} | "
        f"Review status: {selected_record['review_status']}"
    )
    st.info(normalize_reasoning(agent_result["strategic_reasoning"]))
    render_price_trace(agent_result)

# ==========================================
# PAGE 2: MARKET PERFORMANCE (Live Status)
# ==========================================
elif app_mode == "📈 Market Performance":
    st.subheader("Real-Time Market & Baseline Strategy")
    
    live_data = load_live_market_data()
    metrics_df = load_model_metrics()

    c1, c2, c3, c4 = st.columns(4)
    total_otb = sum(v.get("current_otb", 0) for v in live_data.values())
    c1.metric("Demo As-Of Date", DATA_END_DATE.strftime("%Y-%m-%d"))
    c2.metric("30-Day OTB Rooms", f"{total_otb:,}")
    if not metrics_df.empty:
        best = metrics_df.sort_values(["WAPE", "RMSE"]).iloc[0]
        if "Horizon" in best.index:
            best_label = f"{humanize_label(best['Model'])} / {int(best['Horizon'])}d"
        elif "Strategy" in best.index:
            best_label = humanize_label(best["Model"])
        else:
            best_label = humanize_label(best["Model"])
        c3.metric("Best Model", best_label)
        if "Strategy" in best.index:
            c3.caption(f"Strategy: {humanize_label(best['Strategy'])}")
        c4.metric("Backtest WAPE", f"{best['WAPE']:.1f}%")
    else:
        c3.metric("Best Model", "Run forecast")
        c4.metric("Backtest WAPE", "n/a")

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
            "Live_OTB": f"{live_entry.get('current_otb', 0)}",
            "OTB_Occupancy": f"{live_entry.get('current_otb', 0)/BASE_CAPACITY:.1%}",
            "Forecasted_occupancy":f"{min(round(row['Forecasted_Occupancy']*BASE_CAPACITY),237)}",
            "Comp_Price": f"${live_entry.get('competitor_price', 0):.2f}",
            "Recommended_Price": f"${price:.2f}",
            "Booked_Pace": f"{live_entry.get('gross_pace_index', live_entry.get('booking_velocity', 1.0)):.2f}x",
            "Recent_Pickup": f"{live_entry.get('pickup_trend_index', live_entry.get('booking_velocity', 1.0)):.2f}x",
            "System_Status": live_entry.get('status', 'Normal')
        })

    # Plotting
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_df = pd.DataFrame(live_records)
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Live_OTB'].astype(float), 
                             name="Live Occupancy", line=dict(color='#3B82F6', width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df["Forecasted_occupancy"].astype(float), 
                             name="Forecasted Occupancy", line=dict(color="#11F333", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Recommended_Price'].str.lstrip('$').astype(float), 
                             name="Recommended Price ($)", line=dict(color="#F30C9A", width=3)), secondary_y=True)

    fig.update_layout(title="Live Market Snapshot", hovermode="x unified")
    # Left Axis: Number of Rooms
    fig.update_yaxes(
        title_text="<b>Inventory (No. of Rooms)</b>", 
        secondary_y=False,
        gridcolor='LightGray'
    )

    # Right Axis: Price in $
    fig.update_yaxes(
        title_text="<b>Price ($)</b>", 
        secondary_y=True,
        tickprefix="$", 
        showgrid=False # Keep it clean by only showing one set of gridlines
    )
    st.plotly_chart(fig, use_container_width=True)

    # Strategy Table
    st.subheader("📅 Live 30-Day Strategy Feed")
    st.dataframe(pd.DataFrame(live_records), use_container_width=True, height=400, hide_index=True)

    if not metrics_df.empty:
        st.subheader("Forecast Model Backtesting")
        st.dataframe(metrics_df, use_container_width=True, height=260, hide_index=True)

# ==========================================
# PAGE 2: AGENTIC SIMULATION (The "Thinking" View)
# ==========================================
else:
    st.subheader("🤖 Agentic Pricing Simulation")
    
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
        f"Current Live OTB: {current_state['current_otb']} | "
        f"Comp Set: ${current_state.get('comp_low', current_state['competitor_price']):.2f}"
        f" / ${current_state.get('comp_median', current_state['competitor_price']):.2f}"
        f" / ${current_state.get('comp_high', current_state['competitor_price']):.2f}"
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

    if st.sidebar.button("Execute Agentic Decision", type="primary"):
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
        
        # Dashboard Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final ADR", f"${agent_result['final_adr']:.2f}")
        c2.metric("Optimizer ADR", f"${agent_result.get('optimized_price', agent_result['rule_based_price']):.2f}")
        c3.metric("ADR vs Reference", f"{agent_result.get('pct_delta_from_reference', 0):+.2f}%", f"${agent_result.get('absolute_delta', 0):+.2f}")
        c4.metric("Market Gap", f"{agent_result.get('competitor_gap_pct', 0):+.2f}%")
        c5.metric("Pricing Pace", f"{agent_result['pricing_pace_index']}x")

        st.markdown("---")
        
        # Strategic Reasoner
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
        # st.success(agent_result['strategic_reasoning'])
        st.info(normalize_reasoning(agent_result['strategic_reasoning']))
        # Detailed Trace Expander
        with st.expander("Show Technical Node Trace"):
            st.write(f"**Applied Logic Flags:** {agent_result['logic_flags']}")
            st.write(f"**Forecasted Occupancy:** {agent_result['forecasted_occupancy']*100}%")
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

        render_price_trace(agent_result)
