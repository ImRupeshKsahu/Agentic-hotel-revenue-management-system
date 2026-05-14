import json
import math
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from pricing_engine import calculate_recommended_price
from config import API_KEY, CHAT_MODEL, BASE_URL, STRATEGIST_PROMPT_PATH, MIN_PRICE, MAX_PRICE
from openai import OpenAI
from utils.utility_functions import load_prompt
import httpx

http_client = httpx.Client(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)
client = None


def _get_client():
    global client
    if client is None:
        if not API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env before using AI pricing.")
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)
    return client

# 1. Enhanced Agent State
class AgentState(TypedDict):
    target_date: str
    forecasted_occupancy: float
    current_occupancy: float
    competitor_price: float
    demand_shock: float
    manual_demand_shock: float
    local_intel_suggested_shock: float
    local_intel_applied_shock: float
    total_demand_shock: float
    local_intel_estimate: Dict[str, Any]
    manual_event_text: str
    historical_avg_otb: int
    market_state: Dict[str, Any]
    
    # New Pace Variable: 1.0 is normal, >1.0 is fast booking, <1.0 is slow
    booking_velocity: float 
    
    # Internal Logic
    rule_based_price: float
    logic_flags: List[str]
    base_occupancy: float
    pricing_breakdown: Dict[str, Any]
    adjustment_band: Dict[str, Any]
    
    # Agent Output
    final_adr: float
    strategic_reasoning: str
    strategy_applied: str
    perceived_demand_strength: str
    absolute_delta: float
    pct_delta_from_baseline: float
    competitor_gap_pct: float
    guardrails_applied: List[str]
    manual_approval_required: bool
    price_components: List[Dict[str, Any]]


def _target_day_name(target_date: str) -> str:
    try:
        return datetime.fromisoformat(str(target_date)).strftime("%A")
    except ValueError:
        return "Weekday"


def _safe_float(value, fallback=0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return fallback


def _money(value) -> float:
    return round(float(value), 2)


def _pct(value) -> float:
    return round(float(value), 2)


def _find_component(breakdown: Dict[str, Any], driver: str) -> Dict[str, Any]:
    for component in breakdown.get("components", []):
        if component.get("driver") == driver:
            return component
    return {
        "driver": driver,
        "price_before": 0.0,
        "adjustment": 0.0,
        "price_after": 0.0,
        "explanation": "No adjustment was applied.",
    }


def _build_adjustment_band(state: AgentState) -> Dict[str, Any]:
    baseline = _safe_float(state.get("rule_based_price"), MIN_PRICE)
    current_occ = _safe_float(state.get("current_occupancy"))
    forecast_occ = _safe_float(state.get("forecasted_occupancy"))
    velocity = _safe_float(state.get("booking_velocity"), 1.0)
    competitor_price = _safe_float(state.get("competitor_price"), 0.0)

    lower_pct = -0.05
    upper_pct = 0.10
    reasons = ["AI suggestion must remain inside the deterministic safety bounds."]

    if velocity > 1.2 and current_occ > 0.70:
        upper_pct = max(upper_pct, 0.15)
        reasons.append("Fast pickup allows a controlled pace premium up to 15%.")
    if forecast_occ > 0.93:
        upper_pct = max(upper_pct, 0.12)
        reasons.append("Very strong forecast demand allows a modest premium above baseline.")
    if current_occ > 0.95:
        upper_pct = max(upper_pct, 0.20)
        reasons.append("Scarce remaining inventory allows a larger but still bounded premium.")
    if velocity < 0.8 and forecast_occ < 0.70:
        lower_pct = -0.10
        reasons.append("Weak pickup and low forecast allow a controlled discount up to 10%.")

    min_allowed = max(MIN_PRICE, baseline * (1 + lower_pct))
    max_allowed = min(MAX_PRICE, baseline * (1 + upper_pct))

    if competitor_price > 0:
        competitor_ceiling_pct = 1.40 if current_occ > 0.95 else 1.25
        competitor_ceiling = competitor_price * competitor_ceiling_pct
        if max_allowed > competitor_ceiling:
            max_allowed = competitor_ceiling
            reasons.append(f"Competitor-relative ceiling capped unattended AI premium at {competitor_ceiling_pct - 1:.0%} above market.")

    approval_cap = min(baseline * 1.15, baseline + 30)
    if max_allowed > approval_cap:
        max_allowed = approval_cap
        reasons.append("Larger AI premiums require manual approval, so the unattended recommendation is capped.")

    min_allowed = _money(min_allowed)
    max_allowed = _money(max(min_allowed, max_allowed))

    return {
        "min_allowed": min_allowed,
        "max_allowed": max_allowed,
        "lower_pct": _pct(lower_pct * 100),
        "upper_pct": _pct(((max_allowed / baseline) - 1) * 100) if baseline else 0.0,
        "reasons": reasons,
    }


def _validate_ai_price(ai_decision: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    band = state["adjustment_band"]
    baseline = _safe_float(state.get("rule_based_price"), MIN_PRICE)
    raw_price = _safe_float(
        ai_decision.get("final_price", ai_decision.get("override_price", baseline)),
        baseline,
    )
    guardrails = list(band.get("reasons", []))
    local_estimate = state.get("local_intel_estimate") or {}
    guardrails.extend(local_estimate.get("guardrails_applied", []))
    if state.get("manual_event_text") and state.get("local_intel_applied_shock", 0.0) == 0:
        guardrails.append("Local intel was considered as context only and was not included in the baseline price.")

    final_price = min(max(raw_price, band["min_allowed"]), band["max_allowed"])
    final_price = min(max(final_price, MIN_PRICE), MAX_PRICE)
    final_price = _money(final_price)

    if _money(raw_price) != final_price:
        guardrails.append(f"AI suggested ${_money(raw_price):.2f}; validated recommendation was clamped to ${final_price:.2f}.")

    absolute_delta = _money(final_price - baseline)
    pct_delta = _pct((absolute_delta / baseline) * 100) if baseline else 0.0
    competitor_price = _safe_float(state.get("competitor_price"), 0.0)
    competitor_gap = _pct(((final_price - competitor_price) / competitor_price) * 100) if competitor_price else 0.0
    manual_approval_required = abs(pct_delta) > 15 or abs(absolute_delta) > 30

    return {
        "final_adr": final_price,
        "absolute_delta": absolute_delta,
        "pct_delta_from_baseline": pct_delta,
        "competitor_gap_pct": competitor_gap,
        "guardrails_applied": guardrails,
        "manual_approval_required": manual_approval_required,
    }


def _build_price_components(state: AgentState, final_adr: float) -> List[Dict[str, Any]]:
    breakdown = state.get("pricing_breakdown", {})
    baseline = _safe_float(state.get("rule_based_price"), MIN_PRICE)
    pace_delta = _money(final_adr - baseline)

    rows = []
    for driver in ["Base rate", "Occupancy lift", "Manual demand adjustment", "Local intel adjustment", "Day-of-week effect", "Competitor cap"]:
        component = _find_component(breakdown, driver)
        rows.append({
            "Driver": component["driver"],
            "Adjustment": f"${component['adjustment']:+.2f}",
            "Price After": f"${component['price_after']:.2f}",
            "Why": component["explanation"],
        })

    rows.append({
        "Driver": "Pace premium",
        "Adjustment": f"${pace_delta:+.2f}",
        "Price After": f"${final_adr:.2f}",
        "Why": "AI-reviewed strategy adjustment after guardrails; manual and local-intel demand are not counted again here.",
    })
    rows.append({
        "Driver": "Final recommendation",
        "Adjustment": "$+0.00",
        "Price After": f"${final_adr:.2f}",
        "Why": "Final ADR after deterministic rules, AI review, and external guardrails.",
    })
    return rows


def data_ingestion_node(state: AgentState):
    """Normalize market context already loaded by Streamlit.

    Interactive latency stays low because PMS-derived OTB snapshots are prepared
    by the forecast/simulator workflows, not recalculated inside this click path.
    """
    date_data = state.get("market_state") or {}
    velocity = float(date_data.get("booking_velocity", state.get("booking_velocity", 1.0)))
    competitor_price = float(date_data.get("competitor_price", state.get("competitor_price", 120.0)))

    return {
        "booking_velocity": velocity,
        "competitor_price": competitor_price,
    }

# --- NODE 1: THE RULES EXPERT ---
def rules_expert_node(state: AgentState):
    base_occ = max(state['forecasted_occupancy'], state["current_occupancy"])
    manual_shock = _safe_float(state.get("manual_demand_shock"), _safe_float(state.get("demand_shock"), 0.0))
    local_intel_shock = _safe_float(state.get("local_intel_applied_shock"), 0.0)
    total_shock = manual_shock + local_intel_shock
    unclamped_occ = base_occ + total_shock
    sim_occ = max(0, min(1, unclamped_occ))
    day_name = _target_day_name(state["target_date"])
    price, flags, breakdown = calculate_recommended_price(
        occupancy=sim_occ,
        day_name=day_name,
        target_date= state["target_date"],
        competitor_price=state['competitor_price'],
        return_breakdown=True,
        pre_shock_occupancy=base_occ,
        manual_shock=manual_shock,
        local_intel_shock=local_intel_shock,
    )
    if manual_shock != 0:
        flags.append(f"Manual demand adjustment applied ({manual_shock * 100:+.0f}%)")
    local_estimate = state.get("local_intel_estimate") or {}
    if local_intel_shock != 0:
        flags.append(f"Local intel adjustment applied ({local_intel_shock * 100:+.0f}%)")
    elif state.get("manual_event_text"):
        flags.append("Local intel considered as context only")
    if unclamped_occ != sim_occ:
        flags.append("Total demand adjustment was clamped to keep priced occupancy between 0% and 100%")
    if state.get("manual_event_text"):
        flags.append("Local intel context supplied")
    return {
        "rule_based_price": price,
        "logic_flags": flags,
        "base_occupancy": sim_occ,
        "pricing_breakdown": breakdown,
        "manual_demand_shock": manual_shock,
        "local_intel_suggested_shock": _safe_float(local_estimate.get("suggested_shock"), state.get("local_intel_suggested_shock", 0.0)),
        "local_intel_applied_shock": local_intel_shock,
        "total_demand_shock": total_shock,
    }

# --- NODE 2: THE PACE ANALYST ---
def pace_analyst_node(state: AgentState):
    """
    Evaluates how fast we are filling up. 
    Velocity > 1.2 means we are 'AHEAD' of pace.
    Velocity < 0.8 means we are 'BEHIND' pace.
    """
    velocity = state.get('booking_velocity', 1.0)
    pace_status = "Normal"
    
    if velocity > 1.2:
        pace_status = "Aggressive Pickup (Ahead of Pace)"
    elif velocity < 0.8:
        pace_status = "Sluggish Demand (Behind Pace)"
        
    # We add this finding to the logic flags for the LLM to see
    new_flags = state['logic_flags'] + [f"Booking Pace Status: {pace_status}"]
    
    return {"logic_flags": new_flags}

# --- NODE 3: THE AI STRATEGIST ---
def strategist_node(state: AgentState):
    pace_info = next((f for f in state['logic_flags'] if "Pace" in f), "Pace: Normal")
    non_pace_flags = [f for f in state["logic_flags"] if "Pace" not in f]
    adjustment_band = _build_adjustment_band(state)
    state["adjustment_band"] = adjustment_band

    system_message = (
        "You are a Senior Hotel Revenue Strategy Reviewer. The deterministic pricing engine owns the baseline and guardrails. "
        "Your job is to recommend a small, auditable strategy adjustment inside the provided band and explain it clearly. "
        "You must output ONLY a valid JSON object. Do not include markdown formatting or thinking text."
    )
    
    prompt_data = {"current_occ":state['current_occupancy'] * 100,
                   "forecasted_occ":state['forecasted_occupancy'] * 100,
                   "inventory_status":state['current_occupancy']/max(state['forecasted_occupancy'], 0.01),
                   "rule_based_price":state['rule_based_price'],
                   "competitor_price":state['competitor_price'],
                   "booking_velocity":state['booking_velocity'],
                   "manual_demand_shock":state.get('manual_demand_shock', state.get('demand_shock', 0.0))* 100,
                   "local_intel_suggested_shock":state.get('local_intel_suggested_shock', 0.0)* 100,
                   "local_intel_applied_shock":state.get('local_intel_applied_shock', 0.0)* 100,
                   "local_intel_applied_label":"Yes" if state.get('local_intel_applied_shock', 0.0) != 0 else "No",
                   "total_demand_shock":state.get('total_demand_shock', state.get('demand_shock', 0.0))* 100,
                   "local_intel_estimate": json.dumps(state.get("local_intel_estimate", {})),
                   "manual_event_text": state.get("manual_event_text", ""),
                   "pace":", ".join(non_pace_flags) if non_pace_flags else "No rule flags",
                   "pace_info":pace_info,
                   "adjustment_min": adjustment_band["min_allowed"],
                   "adjustment_max": adjustment_band["max_allowed"],
                   "pricing_components": json.dumps(state.get("pricing_breakdown", {}).get("components", []))
                   }
    
    user_prompt = load_prompt(STRATEGIST_PROMPT_PATH ,**prompt_data)

    try:
        # API Call
        response = _get_client().chat.completions.create(
            model= CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=0.2  # Low temperature for consistent financial logic
        )

        ai_decision = json.loads(response.choices[0].message.content)
        validated = _validate_ai_price(ai_decision, state)
        final_adr = validated["final_adr"]
        owner_summary = ai_decision.get("owner_summary") or ai_decision.get("reasoning") or "The baseline price was reviewed against demand, pace, market rate, and safety guardrails."
        if state.get("manual_event_text") and state.get("local_intel_applied_shock", 0.0) == 0:
            owner_summary = "Local intel was considered as context only and was not included in the baseline price. " + owner_summary

        result = {
            **validated,
            "strategy_applied": ai_decision.get("strategy_applied", "Guardrailed AI Review"),
            "perceived_demand_strength": ai_decision.get("perceived_demand_strength", "Medium"),
            "strategic_reasoning": owner_summary,
            "adjustment_band": adjustment_band,
            "price_components": _build_price_components(state, final_adr),
            "adjustment_components": ai_decision.get("adjustment_components", []),
            "local_intel_estimate": state.get("local_intel_estimate", {}),
            "manual_demand_shock": state.get("manual_demand_shock", state.get("demand_shock", 0.0)),
            "local_intel_suggested_shock": state.get("local_intel_suggested_shock", 0.0),
            "local_intel_applied_shock": state.get("local_intel_applied_shock", 0.0),
            "total_demand_shock": state.get("total_demand_shock", state.get("demand_shock", 0.0)),
        }
        return result

    except Exception as e:
        # Robust Fallback to Rules
        final_adr = state['rule_based_price']
        return {
            "final_adr": final_adr,
            "absolute_delta": 0.0,
            "pct_delta_from_baseline": 0.0,
            "competitor_gap_pct": _pct(((final_adr - state['competitor_price']) / state['competitor_price']) * 100) if state.get("competitor_price") else 0.0,
            "guardrails_applied": [f"Strategist Error: {str(e)}. Fallback to rule engine."],
            "manual_approval_required": False,
            "strategy_applied": "Rule-Based Fallback",
            "perceived_demand_strength": "Medium",
            "strategic_reasoning": f"Strategist Error: {str(e)}. Fallback to rule engine.",
            "adjustment_band": adjustment_band,
            "price_components": _build_price_components(state, final_adr),
            "adjustment_components": [],
            "local_intel_estimate": state.get("local_intel_estimate", {}),
            "manual_demand_shock": state.get("manual_demand_shock", state.get("demand_shock", 0.0)),
            "local_intel_suggested_shock": state.get("local_intel_suggested_shock", 0.0),
            "local_intel_applied_shock": state.get("local_intel_applied_shock", 0.0),
            "total_demand_shock": state.get("total_demand_shock", state.get("demand_shock", 0.0)),
        }

# --- GRAPH ORCHESTRATION ---
@lru_cache(maxsize=1)
def create_pricing_agent():
    builder = StateGraph(AgentState)

    builder.add_node("ingests_data", data_ingestion_node)
    builder.add_node("apply_rules", rules_expert_node)
    builder.add_node("analyze_pace", pace_analyst_node)
    builder.add_node("ai_strategist", strategist_node)

    builder.set_entry_point("ingests_data")
    builder.add_edge("ingests_data","apply_rules")
    builder.add_edge("apply_rules", "analyze_pace")
    builder.add_edge("analyze_pace", "ai_strategist")
    builder.add_edge("ai_strategist", END)

    return builder.compile()

# Execution Wrapper
def run_agentic_pricing(
    target_date,
    current_occupancy,
    forecasted_occupancy,
    shock,
    manual_event_text="",
    competitor_price=120.0,
    booking_velocity=1.0,
    historical_avg_otb=1,
    market_state=None,
    manual_demand_shock=None,
    local_intel_estimate=None,
    local_intel_applied_shock=0.0,
):
    manual_shock = shock if manual_demand_shock is None else manual_demand_shock
    local_estimate = local_intel_estimate or {}
    local_suggested_shock = _safe_float(local_estimate.get("suggested_shock"), 0.0)
    local_applied_shock = _safe_float(local_intel_applied_shock, 0.0)
    total_shock = _safe_float(manual_shock, 0.0) + local_applied_shock
    agent = create_pricing_agent()
    result = agent.invoke({
        "target_date": target_date,
        "forecasted_occupancy": forecasted_occupancy,
        "current_occupancy":current_occupancy,
        "competitor_price": competitor_price,
        "booking_velocity": booking_velocity,
        "historical_avg_otb": historical_avg_otb,
        "market_state": market_state or {},
        "demand_shock": total_shock,
        "manual_demand_shock": manual_shock,
        "local_intel_suggested_shock": local_suggested_shock,
        "local_intel_applied_shock": local_applied_shock,
        "total_demand_shock": total_shock,
        "local_intel_estimate": local_estimate,
        "manual_event_text": manual_event_text
    })
    return result

if __name__ == "__main__":
    result = run_agentic_pricing(target_date="2017-09-23",current_occupancy=0.958,forecasted_occupancy=1,shock=0.0)
    print(result)
