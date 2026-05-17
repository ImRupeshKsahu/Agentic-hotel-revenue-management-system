import json
import math
import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from pricing_engine import calculate_recommended_price, normalize_market_context
from config import API_KEY, CHAT_MODEL, BASE_URL, STRATEGIST_PROMPT_PATH, MIN_PRICE, MAX_PRICE, BASE_PRICE, PRICING_DECISION_LOG_PATH
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
        api_key = _resolve_api_key()
        if not api_key:
            raise RuntimeError("AI advisory is unavailable because DEEPSEEK_API_KEY is not configured.")
        client = OpenAI(api_key=api_key, base_url=BASE_URL, http_client=http_client)
    return client


def _resolve_api_key() -> str:
    key = (API_KEY or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if key and key != "your_deepseek_api_key_here":
        return key

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                if line.strip().startswith("DEEPSEEK_API_KEY="):
                    _, value = line.split("=", 1)
                    key = value.strip().strip('"').strip("'")
                    if key and key != "your_deepseek_api_key_here":
                        return key
    except OSError:
        pass

    return ""

# 1. Enhanced Agent State
class AgentState(TypedDict):
    target_date: str
    forecasted_occupancy: float
    current_occupancy: float
    raw_otb_occupancy: float
    adjusted_otb_occupancy: float
    expected_cancellations: float
    sold_out: bool
    competitor_price: float
    market_context: Dict[str, Any]
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
    optimized_price: float
    optimizer_price: float
    optimizer_diagnostics: Dict[str, Any]
    rule_based_price: float
    logic_flags: List[str]
    base_occupancy: float
    pricing_breakdown: Dict[str, Any]
    
    # Advisory + Final Output
    final_adr: float
    strategic_reasoning: str
    strategy_applied: str
    ai_recommended_action: str
    ai_risk_level: str
    ai_review_flags: List[str]
    ai_owner_summary: str
    perceived_demand_strength: str
    absolute_delta: float
    pct_delta_from_reference: float
    pct_delta_from_baseline: float
    competitor_gap_pct: float
    guardrails_applied: List[str]
    manual_approval_required: bool
    price_components: List[Dict[str, Any]]
    price_path_components: List[Dict[str, Any]]
    decision_context_components: List[Dict[str, Any]]


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


TECHNICAL_SUMMARY_TERMS = [
    "blended reference price",
    "candidate optimizer",
    "expected revenue among candidates",
    "elasticity",
    "diagnostics",
    "algorithm",
    "model math",
]


def _sentence_count(text: str) -> int:
    without_decimal_points = re.sub(r"(?<=\d)\.(?=\d)", "", text or "")
    return len([part for part in re.split(r"[.!?]+", without_decimal_points) if part.strip()])


def _needs_manager_rewrite(summary: str, state: AgentState | None = None) -> bool:
    normalized = (summary or "").lower()
    if state:
        raw_booked_occupancy = _safe_float(
            state.get("raw_otb_occupancy"),
            _safe_float(state.get("current_occupancy")),
        )
        retained_otb_occupancy = _safe_float(
            state.get("adjusted_otb_occupancy"),
            _safe_float(state.get("current_occupancy")),
        )
        if raw_booked_occupancy >= 0.9999 or abs(raw_booked_occupancy - retained_otb_occupancy) >= 0.05:
            return True
    return (
        not normalized.strip()
        or any(term in normalized for term in TECHNICAL_SUMMARY_TERMS)
        or _sentence_count(summary) > 3
    )


def _pace_phrase(booking_velocity: float) -> str:
    if booking_velocity >= 1.20:
        return f"pickup is running {round((booking_velocity - 1) * 100)}% ahead of normal pace"
    if booking_velocity <= 0.80:
        return f"pickup is running {round((1 - booking_velocity) * 100)}% behind normal pace"
    return "pickup is close to normal pace"


def _manager_summary(state: AgentState, action: str) -> str:
    adr = _safe_float(state.get("optimized_price"), MIN_PRICE)
    raw_booked_occupancy = _safe_float(
        state.get("raw_otb_occupancy"),
        _safe_float(state.get("current_occupancy")),
    )
    retained_otb_occupancy = _safe_float(
        state.get("adjusted_otb_occupancy"),
        _safe_float(state.get("current_occupancy")),
    )
    forecasted_occupancy = _safe_float(state.get("forecasted_occupancy"))
    booking_velocity = _safe_float(state.get("booking_velocity"), 1.0)
    market_context = state.get("market_context") or {}
    competitor_price = _safe_float(market_context.get("comp_median"), _safe_float(state.get("competitor_price"), 0.0))
    competitor_phrase = (
        f"Competitors are priced at ${competitor_price:.2f}, so this ADR is still below the market."
        if competitor_price and adr < competitor_price
        else f"Competitors are priced at ${competitor_price:.2f}, so this ADR is aligned with the market."
        if competitor_price and abs(adr - competitor_price) < 0.01
        else f"Competitors are priced at ${competitor_price:.2f}, so confirm the premium fits your positioning."
        if competitor_price
        else "Competitor pricing is unavailable, so treat this as an internal demand-led recommendation."
    )
    action_sentence = {
        "Accept Optimizer Price": "Accept the optimizer price if the remaining-room strategy is unchanged.",
        "Review Before Publishing": "Review before publishing to confirm the ADR fits your positioning and remaining-room strategy.",
        "Hold For Manual Approval": "Hold for manual approval before publishing because the risk level is elevated.",
        "Investigate Data Quality": "Investigate the data quality before publishing this ADR.",
    }.get(action, "Review before publishing to confirm the ADR fits your positioning and remaining-room strategy.")

    occupancy_context = f"the hotel is currently {raw_booked_occupancy * 100:.1f}% booked"
    if abs(raw_booked_occupancy - retained_otb_occupancy) >= 0.05:
        occupancy_context += (
            f"; after expected cancellations, retained OTB is {retained_otb_occupancy * 100:.1f}%"
        )
    if abs(raw_booked_occupancy - forecasted_occupancy) >= 0.01:
        occupancy_context += f", while the forecast is {forecasted_occupancy * 100:.1f}%"

    return (
        f"The recommended ADR is ${adr:.2f} because {occupancy_context}. "
        f"{_pace_phrase(booking_velocity).capitalize()}, and {competitor_phrase[0].lower() + competitor_phrase[1:]} "
        f"{action_sentence}"
    )


def _manager_friendly_flags(flags) -> List[str]:
    cleaned = []
    replacements = {
        "blended reference price": "usual pricing level",
        "reference price": "usual pricing level",
        "optimizer": "system",
        "diagnostics": "data",
        "elasticity": "price sensitivity",
    }
    for flag in flags or []:
        text = str(flag).strip()
        if not text:
            continue
        for old, new in replacements.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)
        cleaned.append(text[:140])
    return cleaned[:3]


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


def _format_component_row(component: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Driver": component["driver"],
        "Adjustment": f"${component['adjustment']:+.2f}",
        "Price After": f"${component['price_after']:.2f}",
        "Why": component["explanation"],
    }


def _build_price_path_components(state: AgentState, final_adr: float) -> List[Dict[str, Any]]:
    breakdown = state.get("pricing_breakdown", {})
    reference_price = _safe_float(breakdown.get("reference_price"), _safe_float(state.get("rule_based_price"), MIN_PRICE))

    component_drivers = ["Base rate", "Reference price", "Candidate optimization"]
    if breakdown.get("sold_out"):
        component_drivers.append("Sold-out floor")
    component_drivers.append("Final recommendation")
    rows = [
        _format_component_row(_find_component(breakdown, driver))
        for driver in component_drivers
    ]
    rows.append({
        "Driver": "Reference delta",
        "Adjustment": f"${_money(final_adr - reference_price):+.2f}",
        "Price After": f"${final_adr:.2f}",
        "Why": "Final optimizer ADR compared with the blended reference price.",
    })
    return rows


def _build_decision_context_components(state: AgentState, final_adr: float) -> List[Dict[str, Any]]:
    breakdown = state.get("pricing_breakdown", {})
    context_rows = [
        {
            "Signal": "Current booked occupancy",
            "Value": f"{round(_safe_float(breakdown.get('raw_otb_occupancy'), 0.0) * 100)}%",
            "Why it matters": "Raw OTB determines whether inventory is already sold out.",
        },
        {
            "Signal": "Retained OTB after cancellations",
            "Value": f"{round(_safe_float(breakdown.get('adjusted_otb_occupancy'), 0.0) * 100)}%",
            "Why it matters": "Cancellation-adjusted OTB remains the demand signal used in pricing.",
        },
    ]
    for driver in ["Demand anchor", "Competitor signal"]:
        component = _find_component(breakdown, driver)
        context_rows.append({
            "Signal": component["driver"],
            "Value": (
                f"{round(_safe_float(breakdown.get('demand_anchor'), 0.0) * 100)}%"
                if driver == "Demand anchor"
                else (
                    f"${_safe_float(breakdown.get('competitor_price'), 0.0):.2f}"
                    if breakdown.get("competitor_price") is not None
                    else "Unavailable"
                )
            ),
            "Why it matters": component["explanation"],
        })

    context_rows.append({
        "Signal": "Market premium headroom",
        "Value": f"{round(_safe_float(breakdown.get('allowed_premium_pct'), 0.0) * 100)}%",
        "Why it matters": (
            f"{str(breakdown.get('market_position_regime', 'unknown')).replace('_', ' ').title()} "
            f"with compression score {_safe_float(breakdown.get('compression_score'), 0.0):.2f}."
        ),
    })
    context_rows.append({
        "Signal": "AI advisory",
        "Value": "Review only",
        "Why it matters": "AI reviewed risks and explanation only; it did not change the optimizer price.",
    })
    return context_rows


def data_ingestion_node(state: AgentState):
    """Normalize market context already loaded by Streamlit.

    Interactive latency stays low because PMS-derived OTB snapshots are prepared
    by the forecast/simulator workflows, not recalculated inside this click path.
    """
    date_data = state.get("market_state") or {}
    velocity = float(date_data.get("booking_velocity", state.get("booking_velocity", 1.0)))
    raw_market_context = {
        "comp_low": date_data.get("comp_low"),
        "comp_median": date_data.get("comp_median"),
        "comp_high": date_data.get("comp_high"),
        "sample_size": date_data.get("sample_size"),
        "source_quality": date_data.get("source_quality"),
        "market_regime": date_data.get("market_regime"),
        "market_as_of_timestamp": date_data.get("market_as_of_timestamp"),
    }
    if not any(value is not None for value in raw_market_context.values()):
        raw_market_context = state.get("market_context") or {}
    market_context = normalize_market_context(
        raw_market_context,
        date_data.get("competitor_price", state.get("competitor_price", 120.0)),
    )
    competitor_price = float(market_context.get("comp_median") or state.get("competitor_price", 120.0))
    raw_otb_occupancy = _safe_float(date_data.get("raw_otb_occupancy"), _safe_float(state.get("raw_otb_occupancy"), 0.0))
    adjusted_otb_occupancy = _safe_float(
        date_data.get("adjusted_otb_occupancy"),
        _safe_float(state.get("adjusted_otb_occupancy"), state.get("current_occupancy", 0.0)),
    )
    expected_cancellations = _safe_float(
        date_data.get("expected_cancellations"),
        _safe_float(state.get("expected_cancellations"), 0.0),
    )

    return {
        "booking_velocity": velocity,
        "competitor_price": competitor_price,
        "market_context": market_context,
        "raw_otb_occupancy": raw_otb_occupancy,
        "adjusted_otb_occupancy": adjusted_otb_occupancy,
        "expected_cancellations": expected_cancellations,
        "sold_out": raw_otb_occupancy >= 0.9999,
    }

# --- NODE 1: THE OPTIMIZER ---
def optimizer_node(state: AgentState):
    adjusted_otb_occupancy = _safe_float(state.get("adjusted_otb_occupancy"), state["current_occupancy"])
    base_occ = max(state['forecasted_occupancy'], adjusted_otb_occupancy)
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
        market_context=state.get("market_context"),
        return_breakdown=True,
        pre_shock_occupancy=base_occ,
        manual_shock=manual_shock,
        local_intel_shock=local_intel_shock,
        booking_velocity=state.get("booking_velocity", 1.0),
        manual_event_text=state.get("manual_event_text", ""),
        raw_otb_occupancy=state.get("raw_otb_occupancy"),
        adjusted_otb_occupancy=adjusted_otb_occupancy,
        expected_cancellations=state.get("expected_cancellations", 0.0),
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
    raw_otb_occupancy = _safe_float(state.get("raw_otb_occupancy"), adjusted_otb_occupancy)
    expected_cancellations = _safe_float(state.get("expected_cancellations"), 0.0)
    sold_out = bool(breakdown.get("sold_out"))
    if sold_out:
        flags.append("Sold-out compression regime active; raw OTB reached capacity.")
    if adjusted_otb_occupancy < raw_otb_occupancy:
        flags.append(
            f"Cancellation-adjusted OTB applied ({raw_otb_occupancy * 100:.1f}% raw -> {adjusted_otb_occupancy * 100:.1f}% retained; "
            f"{expected_cancellations:.2f} expected room cancellations removed)"
        )
    optimizer_diagnostics = {
        "reference_price": breakdown.get("reference_price"),
        "raw_otb_occupancy": breakdown.get("raw_otb_occupancy"),
        "adjusted_otb_occupancy": breakdown.get("adjusted_otb_occupancy"),
        "expected_cancellations": breakdown.get("expected_cancellations"),
        "sold_out": breakdown.get("sold_out"),
        "pricing_regime": breakdown.get("pricing_regime"),
        "sold_out_floor_price": breakdown.get("sold_out_floor_price"),
        "sold_out_floor_applied": breakdown.get("sold_out_floor_applied"),
        "material_retention_gap": breakdown.get("material_retention_gap"),
        "pricing_occupancy": breakdown.get("pricing_occupancy"),
        "demand_anchor": breakdown.get("demand_anchor"),
        "elasticity": breakdown.get("elasticity"),
        "market_context": breakdown.get("market_context"),
        "compression_score": breakdown.get("compression_score"),
        "allowed_premium_pct": breakdown.get("allowed_premium_pct"),
        "market_position_regime": breakdown.get("market_position_regime"),
        "comp_median_gap_pct": breakdown.get("comp_median_gap_pct"),
        "comp_high_gap_pct": breakdown.get("comp_high_gap_pct"),
        "selected_candidate": breakdown.get("selected_candidate"),
        "expected_rooms": breakdown.get("expected_rooms"),
        "expected_revenue": breakdown.get("expected_revenue"),
        "competitor_gap_pct": breakdown.get("competitor_gap_pct"),
        "review_flags": breakdown.get("review_flags", []),
        "top_candidates": sorted(
            breakdown.get("optimizer_candidates", []),
            key=lambda row: row.get("expected_revenue", 0),
            reverse=True,
        )[:5],
    }
    return {
        "optimized_price": price,
        "optimizer_price": price,
        "rule_based_price": price,
        "logic_flags": flags,
        "sold_out": sold_out,
        "base_occupancy": sim_occ,
        "pricing_breakdown": breakdown,
        "optimizer_diagnostics": optimizer_diagnostics,
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
    optimized_price = _safe_float(state.get("optimized_price"), MIN_PRICE)
    breakdown = state.get("pricing_breakdown", {})
    diagnostics = state.get("optimizer_diagnostics", {})

    system_message = (
        "You are a Senior Hotel Revenue Strategy Reviewer. The optimizer owns the price. "
        "Your job is to review, explain, and flag risks. You must not change the ADR or propose a replacement price. "
        "You must output ONLY a valid JSON object. Do not include markdown formatting or thinking text."
    )
    
    prompt_data = {"current_booked_occ":state['raw_otb_occupancy'] * 100,
                   "retained_otb_occ":state['adjusted_otb_occupancy'] * 100,
                   "forecasted_occ":state['forecasted_occupancy'] * 100,
                   "inventory_status":state['raw_otb_occupancy']/max(state['forecasted_occupancy'], 0.01),
                   "sold_out_label":"Yes" if state.get("sold_out") else "No",
                   "optimized_price": optimized_price,
                   "reference_price": _safe_float(breakdown.get("reference_price"), BASE_PRICE),
                   "expected_rooms": _safe_float(diagnostics.get("expected_rooms"), 0.0),
                   "expected_revenue": _safe_float(diagnostics.get("expected_revenue"), 0.0),
                   "competitor_price":state['competitor_price'],
                   "comp_low": _safe_float((state.get("market_context") or {}).get("comp_low"), 0.0),
                   "comp_median": _safe_float((state.get("market_context") or {}).get("comp_median"), 0.0),
                   "comp_high": _safe_float((state.get("market_context") or {}).get("comp_high"), 0.0),
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
                   "pricing_components": json.dumps(breakdown.get("components", [])),
                   "optimizer_diagnostics": json.dumps(diagnostics),
                   "review_flags": json.dumps(diagnostics.get("review_flags", [])),
                   }
    
    user_prompt = load_prompt(STRATEGIST_PROMPT_PATH ,**prompt_data)

    try:
        if not _resolve_api_key():
            advisory_message = "AI advisory is unavailable because no DeepSeek API key is configured. Optimizer price retained."
            return {
                "ai_recommended_action": "Review Before Publishing",
                "ai_risk_level": "Medium",
                "ai_review_flags": [advisory_message],
                "ai_owner_summary": advisory_message,
                "strategy_applied": "Optimizer Only",
                "perceived_demand_strength": "Medium",
                "strategic_reasoning": advisory_message,
                "adjustment_components": [],
            }

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
        action = ai_decision.get("ai_recommended_action") or ai_decision.get("recommended_action") or "Accept Optimizer Price"
        if action not in {"Accept Optimizer Price", "Review Before Publishing", "Hold For Manual Approval", "Investigate Data Quality"}:
            action = "Review Before Publishing"
        risk_level = ai_decision.get("ai_risk_level") or ai_decision.get("risk_level") or "Medium"
        if risk_level not in {"Low", "Medium", "High"}:
            risk_level = "Medium"
        review_flags = ai_decision.get("ai_review_flags") or ai_decision.get("review_flags") or []
        if not isinstance(review_flags, list):
            review_flags = [str(review_flags)]
        review_flags = _manager_friendly_flags(review_flags)
        owner_summary = ai_decision.get("ai_owner_summary") or ai_decision.get("owner_summary") or "The optimizer price was reviewed against demand, pace, market rate, and safety guardrails."
        if state.get("manual_event_text") and state.get("local_intel_applied_shock", 0.0) == 0:
            owner_summary = "Local intel was considered as context only and was not included in the baseline price. " + owner_summary
        if _needs_manager_rewrite(owner_summary, state):
            owner_summary = _manager_summary(state, action)

        return {
            "ai_recommended_action": action,
            "ai_risk_level": risk_level,
            "ai_review_flags": review_flags,
            "ai_owner_summary": owner_summary,
            "strategy_applied": action,
            "perceived_demand_strength": ai_decision.get("perceived_demand_strength", risk_level),
            "strategic_reasoning": owner_summary,
            "adjustment_components": ai_decision.get("adjustment_components", []),
        }

    except Exception as e:
        fallback_reason = f"AI advisory unavailable: {str(e)} Optimizer price retained."
        return {
            "ai_recommended_action": "Review Before Publishing",
            "ai_risk_level": "Medium",
            "ai_review_flags": [fallback_reason],
            "ai_owner_summary": fallback_reason,
            "strategy_applied": "Optimizer With AI Fallback",
            "perceived_demand_strength": "Medium",
            "strategic_reasoning": fallback_reason,
            "adjustment_components": [],
        }


# --- NODE 4: DETERMINISTIC VALIDATION ---
def validation_node(state: AgentState):
    final_adr = _money(min(max(_safe_float(state.get("optimized_price"), MIN_PRICE), MIN_PRICE), MAX_PRICE))
    breakdown = state.get("pricing_breakdown", {})
    diagnostics = state.get("optimizer_diagnostics", {})
    reference_price = _safe_float(breakdown.get("reference_price"), BASE_PRICE)
    competitor_price = _safe_float(state.get("competitor_price"), 0.0)
    absolute_delta = _money(final_adr - reference_price)
    pct_delta = _pct((absolute_delta / reference_price) * 100) if reference_price else 0.0
    competitor_gap = _pct(((final_adr - competitor_price) / competitor_price) * 100) if competitor_price else 0.0

    deterministic_flags = list(diagnostics.get("review_flags", []))
    local_estimate = state.get("local_intel_estimate") or {}
    deterministic_flags.extend(local_estimate.get("guardrails_applied", []))
    ai_flags = list(state.get("ai_review_flags", []))
    guardrails = deterministic_flags + ai_flags

    action = state.get("ai_recommended_action", "Accept Optimizer Price")
    manual_approval_required = (
        action in {"Hold For Manual Approval", "Investigate Data Quality"}
        or state.get("ai_risk_level") == "High"
        or abs(pct_delta) > 20
        or abs(absolute_delta) > 30
        or any("review" in str(flag).lower() for flag in deterministic_flags)
    )

    owner_summary = state.get("ai_owner_summary") or "Optimizer price retained after advisory review."
    if _needs_manager_rewrite(owner_summary, state):
        owner_summary = _manager_summary(state, action)

    return {
        "final_adr": final_adr,
        "optimized_price": final_adr,
        "optimizer_price": final_adr,
        "rule_based_price": final_adr,
        "absolute_delta": absolute_delta,
        "pct_delta_from_reference": pct_delta,
        "pct_delta_from_baseline": pct_delta,
        "competitor_gap_pct": competitor_gap,
        "guardrails_applied": guardrails,
        "manual_approval_required": manual_approval_required,
        "strategy_applied": action,
        "strategic_reasoning": owner_summary,
        "price_components": _build_price_path_components(state, final_adr),
        "price_path_components": _build_price_path_components(state, final_adr),
        "decision_context_components": _build_decision_context_components(state, final_adr),
        "market_context": state.get("market_context", {}),
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
    builder.add_node("optimize_price", optimizer_node)
    builder.add_node("analyze_pace", pace_analyst_node)
    builder.add_node("ai_strategist", strategist_node)
    builder.add_node("validate_decision", validation_node)

    builder.set_entry_point("ingests_data")
    builder.add_edge("ingests_data","optimize_price")
    builder.add_edge("optimize_price", "analyze_pace")
    builder.add_edge("analyze_pace", "ai_strategist")
    builder.add_edge("ai_strategist", "validate_decision")
    builder.add_edge("validate_decision", END)

    return builder.compile()

# Execution Wrapper
def run_agentic_pricing(
    target_date,
    current_occupancy,
    forecasted_occupancy,
    shock,
    manual_event_text="",
    competitor_price=120.0,
    market_context=None,
    booking_velocity=1.0,
    historical_avg_otb=1,
    market_state=None,
    manual_demand_shock=None,
    local_intel_estimate=None,
    local_intel_applied_shock=0.0,
    raw_otb_occupancy=None,
    adjusted_otb_occupancy=None,
    expected_cancellations=None,
    record_decision=True,
):
    manual_shock = shock if manual_demand_shock is None else manual_demand_shock
    local_estimate = local_intel_estimate or {}
    local_suggested_shock = _safe_float(local_estimate.get("suggested_shock"), 0.0)
    local_applied_shock = _safe_float(local_intel_applied_shock, 0.0)
    total_shock = _safe_float(manual_shock, 0.0) + local_applied_shock
    market_state = market_state or {}
    total_rooms = max(_safe_float(market_state.get("total_rooms"), 0.0), 1.0)
    inferred_raw_otb_occupancy = (
        _safe_float(market_state.get("current_otb"), 0.0) / total_rooms
        if market_state.get("current_otb") is not None
        else current_occupancy
    )
    raw_occ = _safe_float(raw_otb_occupancy, inferred_raw_otb_occupancy)
    adjusted_occ = _safe_float(
        adjusted_otb_occupancy,
        _safe_float(market_state.get("adjusted_otb_occupancy"), current_occupancy),
    )
    expected_cancel_rooms = (
        _safe_float(expected_cancellations)
        if expected_cancellations is not None
        else _safe_float(market_state.get("expected_cancellations"), 0.0)
    )
    resolved_market_context = normalize_market_context(
        market_context or {
            "comp_low": market_state.get("comp_low"),
            "comp_median": market_state.get("comp_median"),
            "comp_high": market_state.get("comp_high"),
            "sample_size": market_state.get("sample_size"),
            "source_quality": market_state.get("source_quality"),
            "market_regime": market_state.get("market_regime"),
            "market_as_of_timestamp": market_state.get("market_as_of_timestamp"),
        },
        competitor_price,
    )
    competitor_price = _safe_float(resolved_market_context.get("comp_median"), competitor_price)
    agent = create_pricing_agent()
    result = agent.invoke({
        "target_date": target_date,
        "forecasted_occupancy": forecasted_occupancy,
        "current_occupancy": adjusted_occ,
        "raw_otb_occupancy": raw_occ,
        "adjusted_otb_occupancy": adjusted_occ,
        "expected_cancellations": expected_cancel_rooms,
        "sold_out": raw_occ >= 0.9999,
        "competitor_price": competitor_price,
        "market_context": resolved_market_context,
        "booking_velocity": booking_velocity,
        "historical_avg_otb": historical_avg_otb,
        "market_state": market_state,
        "demand_shock": total_shock,
        "manual_demand_shock": manual_shock,
        "local_intel_suggested_shock": local_suggested_shock,
        "local_intel_applied_shock": local_applied_shock,
        "total_demand_shock": total_shock,
        "local_intel_estimate": local_estimate,
        "manual_event_text": manual_event_text
    })
    if record_decision:
        _append_pricing_decision_log(result)
    return result


def _append_pricing_decision_log(result: Dict[str, Any]) -> None:
    breakdown = result.get("pricing_breakdown", {})
    diagnostics = result.get("optimizer_diagnostics", {})
    payload = {
        "logged_at": datetime.now(UTC).isoformat(),
        "target_date": result.get("target_date"),
        "forecasted_occupancy": result.get("forecasted_occupancy"),
        "raw_otb_occupancy": result.get("raw_otb_occupancy"),
        "adjusted_otb_occupancy": result.get("adjusted_otb_occupancy"),
        "booking_velocity": result.get("booking_velocity"),
        "market_context": result.get("market_context") or breakdown.get("market_context"),
        "selected_adr": result.get("final_adr"),
        "compression_score": breakdown.get("compression_score"),
        "allowed_premium_pct": breakdown.get("allowed_premium_pct"),
        "market_position_regime": breakdown.get("market_position_regime"),
        "selected_candidate": diagnostics.get("selected_candidate"),
        "top_candidates": diagnostics.get("top_candidates"),
        "observed_rooms_sold": None,
        "observed_revenue": None,
    }
    os.makedirs(os.path.dirname(PRICING_DECISION_LOG_PATH), exist_ok=True)
    with open(PRICING_DECISION_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")

if __name__ == "__main__":
    result = run_agentic_pricing(target_date="2017-09-23",current_occupancy=0.958,forecasted_occupancy=1,shock=0.0)
    print(result)
