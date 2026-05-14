import sys
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from plotly.subplots import make_subplots
from config import BASE_PRICE,BASE_CAPACITY, DATA_END_DATE, FORECAST_OUTPUT_PATH, LIVE_MARKET_STATE_PATH, MODEL_COMPARISON_PATH, OTB_SNAPSHOT_PATH
from pricing_engine import calculate_recommended_price
from pricing_agent import run_agentic_pricing
from local_intel_estimator import estimate_local_intel_impact
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
def load_live_market_state_cached(json_path, otb_path, forecast_path, json_mtime, otb_mtime, forecast_mtime):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    if not os.path.exists(otb_path):
        return {}

    otb_df = pd.read_csv(otb_path)
    otb_df["Date"] = pd.to_datetime(otb_df["Date"])

    if os.path.exists(forecast_path):
        forecast_rates = pd.read_csv(forecast_path)
        forecast_rates["Date"] = pd.to_datetime(forecast_rates["Date"])
        if "Competitor_Rate" in forecast_rates.columns:
            otb_df = otb_df.merge(forecast_rates[["Date", "Competitor_Rate"]], on="Date", how="left")

    state = {}
    for row in otb_df.itertuples(index=False):
        velocity = float(getattr(row, "Booking_Velocity", 1.0))
        status = "Normal"
        if velocity >= 1.2:
            status = "Ahead of historical pace"
        elif velocity <= 0.8:
            status = "Behind historical pace"

        competitor_price = getattr(row, "Competitor_Rate", pd.NA)
        if pd.isna(competitor_price):
            competitor_price = getattr(row, "OTB_ADR", pd.NA)
        if pd.isna(competitor_price):
            competitor_price = 120.0

        state[pd.to_datetime(row.Date).strftime("%Y-%m-%d")] = {
            "current_otb": int(getattr(row, "Live_OTB", 0)),
            "historical_avg_otb": int(getattr(row, "Historical_Avg_OTB", 1)),
            "competitor_price": round(float(competitor_price), 2),
            "total_rooms": int(getattr(row, "Capacity", BASE_CAPACITY)),
            "booking_velocity": velocity,
            "status": status,
        }
    return state


def load_live_market_data():
    return load_live_market_state_cached(
        LIVE_MARKET_STATE_PATH,
        OTB_SNAPSHOT_PATH,
        FORECAST_OUTPUT_PATH,
        file_mtime(LIVE_MARKET_STATE_PATH),
        file_mtime(OTB_SNAPSHOT_PATH),
        file_mtime(FORECAST_OUTPUT_PATH),
    )


def load_model_metrics():
    return load_model_metrics_cached(MODEL_COMPARISON_PATH, file_mtime(MODEL_COMPARISON_PATH))

# 4. Sidebar Navigation
st.sidebar.header("🕹️ System Controls")
app_mode = st.sidebar.radio("Switch View", ["📈 Market Performance", "🤖 Agentic Simulation"])
forecast_df = load_forecast_output_cached(FORECAST_OUTPUT_PATH, file_mtime(FORECAST_OUTPUT_PATH))

# ==========================================
# PAGE 1: MARKET PERFORMANCE (Live Status)
# ==========================================
if app_mode == "📈 Market Performance":
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
            best_label = f"{best['Model']} / {int(best['Horizon'])}d"
        elif "Strategy" in best.index:
            best_label = f"{best['Model']} ({best['Strategy']})"
        else:
            best_label = str(best["Model"])
        c3.metric("Best Model", best_label)
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
            occupancy=max(float(live_entry.get('current_otb', 0)/BASE_CAPACITY), float(row['Forecasted_Occupancy'])),
            day_name=row['Date'].strftime('%A'),
            target_date=d_str,
            competitor_price=live_entry.get('competitor_price', 120.0)
        )
        
        live_records.append({
            "Date": d_str,
            "Day": row['Date'].strftime('%A'),
            "Live_OTB": f"{live_entry.get('current_otb', 0)}",
            "OTB_Occupancy": f"{live_entry.get('current_otb', 0)/BASE_CAPACITY:.1%}",
            "Forecasted_occupancy":f"{min(round(row['Forecasted_Occupancy']*BASE_CAPACITY),237)}",
            "Comp_Price": f"${live_entry.get('competitor_price', 0):.2f}",
            "Recommended_Price": f"${price:.2f}",
            "Booking_Velocity": f"{live_entry.get('booking_velocity', 1.0):.2f}x",
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
    st.dataframe(pd.DataFrame(live_records), use_container_width=True, height=400)

    if not metrics_df.empty:
        st.subheader("Forecast Model Backtesting")
        st.dataframe(metrics_df, use_container_width=True, height=260)

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
            "historical_avg_otb": 1,
            "competitor_price": 120.0,
            "booking_velocity": 1.0,
            "status": "Fallback snapshot",
        },
    )

    st.sidebar.info(f"Current Live OTB: {current_state['current_otb']} | Comp Price: ${current_state['competitor_price']}")
    st.sidebar.subheader("Local Intel")
    manual_event = st.sidebar.text_input("Enter local event (e.g., '100-person wedding block')")

    target_date_parsed = pd.to_datetime(target_date, format="%Y/%m/%d")
    result = forecast_df.loc[forecast_df["Date"] == target_date_parsed, "Forecasted_Occupancy"]
    forecasted_occ = float(result.iloc[0]) if not result.empty else float(current_state['current_otb'] / BASE_CAPACITY)
    current_occ = float(current_state['current_otb'] / BASE_CAPACITY)

    local_intel_estimate = estimate_local_intel_impact(
        manual_event,
        current_occ=current_occ,
        forecast_occ=forecasted_occ,
        booking_velocity=float(current_state.get("booking_velocity", 1.0)),
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
    st.sidebar.caption(f"Total demand shock included in baseline: {total_baseline_shock * 100:+.1f}%")

    if st.sidebar.button("Execute Agentic Decision", type="primary"):
        # --- THE LIVE LOG ---
        log_placeholder = st.empty()
        with log_placeholder.container():
            st.write("🔍 **Node 1: Data Ingestion** - Using cached PMS-derived market snapshot...")
            st.write("📐 **Node 2: Rules Expert** - Applying yield and competitor rules...")
            st.write("📈 **Node 3: Pace Analyst** - Reading prepared booking velocity...")
            st.write("🧠 **Node 4: AI Strategist** - Consulted DeepSeek. Evaluating market anomalies...")

        agent_result = run_agentic_pricing(
            target_date=d_str,
            current_occupancy=current_occ,
            forecasted_occupancy=float(forecasted_occ),
            shock=demand_shock,
            manual_event_text=manual_event,
            competitor_price=float(current_state.get("competitor_price", 120.0)),
            booking_velocity=float(current_state.get("booking_velocity", 1.0)),
            historical_avg_otb=int(current_state.get("historical_avg_otb", 1)),
            market_state=current_state,
            manual_demand_shock=demand_shock,
            local_intel_estimate=local_intel_estimate,
            local_intel_applied_shock=local_intel_applied_shock,
        )

        # Clear logs and show results
        log_placeholder.empty()
        
        # Dashboard Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Agent ADR", f"${agent_result['final_adr']:.2f}")
        c2.metric("Rules Baseline", f"${agent_result['rule_based_price']:.2f}")
        c3.metric("AI Margin", f"{agent_result.get('pct_delta_from_baseline', 0):+.2f}%", f"${agent_result.get('absolute_delta', 0):+.2f}")
        c4.metric("Market Gap", f"{agent_result.get('competitor_gap_pct', 0):+.2f}%")
        c5.metric("Booking Pace", f"{agent_result['booking_velocity']}x")

        st.markdown("---")
        
        # Strategic Reasoner
        st.subheader("🕵️ Agent Reasoning Trace")
        # st.success(agent_result['strategic_reasoning'])
        st.success(normalize_reasoning(agent_result['strategic_reasoning']))
        # Detailed Trace Expander
        with st.expander("Show Technical Node Trace"):
            st.write(f"**Applied Logic Flags:** {agent_result['logic_flags']}")
            st.write(f"**Forecasted Occupancy:** {agent_result['forecasted_occupancy']*100}%")
            st.write(f"**Adjustment Band:** {agent_result.get('adjustment_band', {})}")
            st.write(f"**Local Intel Estimate:** {agent_result.get('local_intel_estimate', {})}")
            st.write(f"**Manual Demand Adjustment:** {agent_result.get('manual_demand_shock', 0) * 100:+.1f}%")
            st.write(f"**Local Intel Applied Adjustment:** {agent_result.get('local_intel_applied_shock', 0) * 100:+.1f}%")
            st.write(f"**Total Baseline Demand Shock:** {agent_result.get('total_demand_shock', 0) * 100:+.1f}%")
            st.write(f"**Guardrails Applied:** {agent_result.get('guardrails_applied', [])}")
            st.write(f"**Manual Approval Required:** {agent_result.get('manual_approval_required', False)}")

        st.subheader("Price Composition Analysis")
        price_components = agent_result.get("price_components", [])
        if price_components:
            st.dataframe(pd.DataFrame(price_components), use_container_width=True, hide_index=True)
        else:
            fallback_rows = [
                {"Driver": "Base rate", "Adjustment": "$+0.00", "Price After": f"${BASE_PRICE:.2f}", "Why": "Starting public rate."},
                {"Driver": "Rule-based baseline", "Adjustment": f"${agent_result['rule_based_price'] - BASE_PRICE:+.2f}", "Price After": f"${agent_result['rule_based_price']:.2f}", "Why": "Deterministic pricing engine output."},
                {"Driver": "AI-reviewed margin", "Adjustment": f"${agent_result['final_adr'] - agent_result['rule_based_price']:+.2f}", "Price After": f"${agent_result['final_adr']:.2f}", "Why": "Validated strategy adjustment."},
            ]
            st.dataframe(pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True)

        guardrail_rows = [{"Guardrail": item} for item in agent_result.get("guardrails_applied", [])]
        if guardrail_rows:
            st.subheader("Guardrail Audit")
            st.dataframe(pd.DataFrame(guardrail_rows), use_container_width=True, hide_index=True)
