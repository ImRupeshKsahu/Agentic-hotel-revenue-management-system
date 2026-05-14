import re
from typing import Any, Dict


MAX_DEFAULT_SHOCK = 0.10
MAX_HIGH_CONFIDENCE_DEMAND_SHOCK = 0.15
MIN_DEFAULT_SHOCK = -0.10


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _has_numbered_group(text: str) -> bool:
    return bool(re.search(r"\b\d{2,4}\s*(-|\s)?(person|people|guest|room|rooms|pax)\b", text))


def _is_major_sports_event(text: str) -> bool:
    sports_markers = [
        "fifa", "world cup", "cup final", "final", "semi final", "semifinal",
        "championship", "football", "cricket", "ipl", "olympics", "grand prix",
        "race", "derby",
    ]
    return _contains_any(text, sports_markers)


def _build_result(
    classification: str,
    suggested_shock: float,
    confidence: str,
    rationale: str,
    guardrails,
    apply_allowed: bool,
) -> Dict[str, Any]:
    capped = max(MIN_DEFAULT_SHOCK, min(MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, suggested_shock))
    guardrail_list = list(guardrails)
    if capped != suggested_shock:
        guardrail_list.append("Suggested local intel impact was capped by estimator guardrails.")

    if confidence == "Low" and capped != 0:
        capped = 0.0
        apply_allowed = False
        guardrail_list.append("Low-confidence local intel cannot be applied to baseline.")

    if capped == 0:
        apply_allowed = False

    return {
        "classification": classification,
        "suggested_shock_pct": round(capped * 100, 1),
        "suggested_shock": round(capped, 4),
        "confidence": confidence,
        "rationale": rationale,
        "guardrails_applied": guardrail_list,
        "apply_allowed": apply_allowed,
    }


def estimate_local_intel_impact(
    text: str,
    current_occ: float,
    forecast_occ: float,
    booking_velocity: float,
) -> Dict[str, Any]:
    """Conservative rule-based local intel estimator.

    The estimator deliberately avoids autonomous pricing decisions. It produces a
    suggested occupancy shock and guardrail narrative; the UI must ask the user
    before applying that suggestion to the baseline.
    """
    normalized = (text or "").strip().lower()
    if not normalized:
        return _build_result(
            classification="Irrelevant",
            suggested_shock=0.0,
            confidence="Low",
            rationale="No local intel was supplied.",
            guardrails=["No local intel impact was estimated."],
            apply_allowed=False,
        )

    current_occ = max(0.0, min(1.0, float(current_occ or 0.0)))
    forecast_occ = max(0.0, min(1.0, float(forecast_occ or 0.0)))
    booking_velocity = float(booking_velocity or 1.0)

    demand_terms = [
        "wedding", "marriage", "banquet", "concert", "conference", "convention",
        "expo", "summit", "festival", "tournament", "match", "event nearby",
        "fifa", "world cup", "cup final", "final", "semi final", "semifinal",
        "championship", "football", "cricket", "ipl", "olympics", "grand prix",
        "race", "derby",
    ]
    sellout_terms = ["sold out", "sold-out", "nearby hotels full", "hotels full", "citywide"]
    disruption_terms = [
        "traffic", "traffic jam", "road closure", "blocked road", "strike",
        "protest", "weather", "storm", "flood", "airport shutdown", "cancelled flight",
        "canceled flight", "entry of the city",
    ]
    stranded_terms = ["stranded", "overnight stay", "last minute rooms", "walk-in", "walk ins"]
    competitor_terms = ["competitor", "comp set", "nearby hotel", "other hotels", "market rate"]

    guardrails = ["Local intel is an estimate and is never applied automatically."]

    has_demand_signal = _contains_any(normalized, demand_terms)
    has_sellout_signal = _contains_any(normalized, sellout_terms)
    is_major_sports_event = _is_major_sports_event(normalized)

    if _contains_any(normalized, disruption_terms):
        if _contains_any(normalized, stranded_terms):
            return _build_result(
                classification="Disruption With Room Demand",
                suggested_shock=0.05,
                confidence="Medium",
                rationale="The disruption may create short-notice room demand from stranded travelers.",
                guardrails=guardrails + ["Disruption-driven demand is capped at +5% unless externally validated."],
                apply_allowed=True,
            )

        if "airport shutdown" in normalized or "cancelled flight" in normalized or "canceled flight" in normalized:
            return _build_result(
                classification="Operational Disruption / Ambiguous Demand",
                suggested_shock=-0.03,
                confidence="Medium",
                rationale="Airport disruption may reduce arrivals unless there is evidence of stranded-room demand.",
                guardrails=guardrails + ["Disruption signals default conservative and require user approval."],
                apply_allowed=True,
            )

        return _build_result(
            classification="Operational Disruption / Ambiguous Demand",
            suggested_shock=0.0,
            confidence="Low",
            rationale="Traffic or access disruption is ambiguous; it may delay arrivals without increasing hotel demand.",
            guardrails=guardrails + ["Traffic, weather, protests, and road closures default to context-only unless room demand is explicit."],
            apply_allowed=False,
        )

    if has_demand_signal:
        if has_sellout_signal:
            return _build_result(
                classification="Event",
                suggested_shock=MAX_HIGH_CONFIDENCE_DEMAND_SHOCK,
                confidence="High",
                rationale="The local event includes strong sell-out language, suggesting high compression demand.",
                guardrails=guardrails + ["High-confidence demand event capped at +15%."],
                apply_allowed=True,
            )

        if is_major_sports_event:
            suggested = 0.05
            if forecast_occ > 0.90 or booking_velocity > 1.2:
                suggested = 0.10
            elif forecast_occ > 0.80 or booking_velocity > 1.1:
                suggested = 0.08
            return _build_result(
                classification="Event",
                suggested_shock=suggested,
                confidence="Medium",
                rationale="Major sports final likely creates local demand compression, but impact is not applied unless approved.",
                guardrails=guardrails + ["Major sports event impact is conservative unless sell-out evidence is supplied."],
                apply_allowed=True,
            )

        if _has_numbered_group(normalized) or _contains_any(normalized, ["wedding", "conference", "convention", "expo", "summit"]):
            suggested = 0.08
            if forecast_occ > 0.90 or booking_velocity > 1.2:
                suggested = 0.10
            return _build_result(
                classification="Event",
                suggested_shock=suggested,
                confidence="Medium",
                rationale="The local intel describes a plausible demand-generating event.",
                guardrails=guardrails + [f"Standard event impact capped at +{int(MAX_DEFAULT_SHOCK * 100)}%."],
                apply_allowed=True,
            )

        return _build_result(
            classification="Event",
            suggested_shock=0.05,
            confidence="Medium",
            rationale="The local intel may generate incremental lodging demand, but details are limited.",
            guardrails=guardrails + ["Limited event details keep the suggested impact conservative."],
            apply_allowed=True,
        )

    if _contains_any(normalized, competitor_terms):
        return _build_result(
            classification="Competitor Signal",
            suggested_shock=0.0,
            confidence="Low",
            rationale="The text appears to describe competitor context rather than direct demand impact.",
            guardrails=guardrails + ["Competitor signals should affect benchmarking, not occupancy shock."],
            apply_allowed=False,
        )

    return _build_result(
        classification="Ambiguous",
        suggested_shock=0.0,
        confidence="Low",
        rationale="The local intel does not clearly indicate a measurable demand change.",
        guardrails=guardrails + ["Ambiguous intel is treated as context only."],
        apply_allowed=False,
    )
