import csv
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import pandas as pd

from config import LOCAL_INTEL_CALENDAR_PATH


MAX_DEFAULT_SHOCK = 0.10
MAX_HIGH_CONFIDENCE_DEMAND_SHOCK = 0.18
MAX_ADR_HEADROOM = 0.20
MIN_DEFAULT_SHOCK = -0.10
HOTEL_LOCATION_PROFILE = "lisbon_city_hotel_anonymous"
PROXIMITY_LABELS = {
    "nearby": "Nearby",
    "city_center_relevant": "City center relevant",
    "citywide": "Citywide",
    "distant_or_uncertain": "Distant or uncertain",
    "not_specified": "Not specified",
}
PROXIMITY_TERMS = {
    "citywide": [
        "citywide", "city-wide", "marketwide", "market-wide", "market wide",
        "across the city", "whole city", "entire city", "all over the city",
        "city sold out", "city compression", "major city event", "destination event",
        "multiple venues", "hotels full", "nearby hotels full", "all hotels full",
    ],
    "nearby": [
        "nearby", "near by", "next door", "next to", "adjacent", "beside",
        "opposite", "across the street", "same block", "around the corner",
        "walking distance", "walkable", "short walk", "close to", "near the hotel",
        "near hotel", "near our hotel", "local venue", "in the area",
    ],
    "city_center_relevant": [
        "city center", "city centre", "downtown", "central", "central lisbon",
        "lisbon center", "lisbon centre", "old town", "main square",
        "historic center", "historic centre", "business district",
        "financial district", "cbd", "central area", "metro area",
    ],
    "distant_or_uncertain": [
        "far away", "far from hotel", "far from the hotel", "outskirts", "suburb",
        "suburban", "outside the city", "outside lisbon", "airport area",
        "remote", "distant", "unclear location", "unknown location",
        "location unknown", "not sure where", "somewhere in lisbon",
    ],
}


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _has_numbered_group(text: str) -> bool:
    return bool(re.search(r"\b\d{2,5}\s*(-|\s)?(person|people|guest|room|rooms|pax|participant|participants|attendee|attendees)\b", text))


def _is_major_sports_event(text: str) -> bool:
    sports_markers = [
        "fifa", "world cup", "cup final", "final", "semi final", "semifinal",
        "championship", "football", "cricket", "ipl", "olympics", "grand prix",
        "race", "derby", "benfica", "sporting", "champions league", "estadio",
        "stadium", "match",
    ]
    return _contains_any(text, sports_markers)


def _safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
        if pd.notna(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def _normalize_date(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return ""


@lru_cache(maxsize=1)
def load_local_intel_calendar(path: str = LOCAL_INTEL_CALENDAR_PATH) -> List[Dict[str, Any]]:
    """Load the transparent Lisbon September 2017 demo event calendar."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        rows = []
        for row in csv.DictReader(csv_file):
            normalized = dict(row)
            normalized["stay_date"] = _normalize_date(normalized.get("stay_date"))
            normalized["expected_attendance"] = _safe_float(normalized.get("expected_attendance"), 0.0)
            rows.append(normalized)
        return rows


def local_intel_events_for_date(target_date, path: str = LOCAL_INTEL_CALENDAR_PATH) -> List[Dict[str, Any]]:
    date_key = _normalize_date(target_date)
    if not date_key:
        return []
    return [row for row in load_local_intel_calendar(path) if row.get("stay_date") == date_key]


def local_intel_summary_for_date(target_date) -> str:
    events = local_intel_events_for_date(target_date)
    if not events:
        return ""
    return "; ".join(
        f"{event.get('event_name')} at {event.get('area_or_venue')}"
        for event in events
        if event.get("event_name")
    )


def _category_profile(category: str, attendance: float) -> Dict[str, Any]:
    category = (category or "").lower()
    if "sports" in category or attendance >= 30000:
        return {
            "classification": "Major Sports Event",
            "base_shock": 0.10,
            "base_headroom": 0.12,
            "intensity": 0.78,
            "confidence": "High" if attendance >= 30000 else "Medium",
        }
    if "festival" in category or "cultural" in category or "music" in category:
        return {
            "classification": "Cultural / Festival Cluster",
            "base_shock": 0.06,
            "base_headroom": 0.06,
            "intensity": 0.55,
            "confidence": "Medium",
        }
    if "conference" in category or "business" in category or "summit" in category or 1000 <= attendance <= 5000:
        return {
            "classification": "Business Event",
            "base_shock": 0.03,
            "base_headroom": 0.04,
            "intensity": 0.35,
            "confidence": "Medium",
        }
    return {
        "classification": "Event",
        "base_shock": 0.05,
        "base_headroom": 0.04,
        "intensity": 0.45,
        "confidence": "Medium",
    }


def _proximity_factor(bucket: str) -> float:
    bucket = (bucket or "").lower()
    if bucket == "nearby":
        return 1.20
    if bucket == "city_center_relevant":
        return 1.00
    if bucket == "citywide":
        return 0.90
    if bucket == "not_specified":
        return 1.00
    return 0.65


def _normalize_proximity_bucket(bucket: str | None) -> str:
    normalized = (bucket or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "city_center": "city_center_relevant",
        "city_centre": "city_center_relevant",
        "central": "city_center_relevant",
        "downtown": "city_center_relevant",
        "city_wide": "citywide",
        "market_wide": "citywide",
        "uncertain": "distant_or_uncertain",
        "distant": "distant_or_uncertain",
        "far": "distant_or_uncertain",
        "unknown": "distant_or_uncertain",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in PROXIMITY_LABELS:
        return normalized
    return ""


def infer_local_intel_proximity(text: str, override_bucket: str | None = None) -> Dict[str, Any]:
    """Infer a transparent proximity bucket for manager-entered local intel."""
    override = _normalize_proximity_bucket(override_bucket)
    if override and override != "not_specified":
        return {
            "bucket": override,
            "label": PROXIMITY_LABELS[override],
            "factor": _proximity_factor(override),
            "source": "manager_override",
            "matched_terms": [],
        }

    normalized = (text or "").lower()
    for bucket, terms in PROXIMITY_TERMS.items():
        matched = [term for term in terms if term in normalized]
        if matched:
            return {
                "bucket": bucket,
                "label": PROXIMITY_LABELS[bucket],
                "factor": _proximity_factor(bucket),
                "source": "text_inferred",
                "matched_terms": matched[:3],
            }

    return {
        "bucket": "not_specified",
        "label": PROXIMITY_LABELS["not_specified"],
        "factor": _proximity_factor("not_specified"),
        "source": "not_specified",
        "matched_terms": [],
    }


def _market_proof_boost(market_context, retained_pace_index: float, pickup_trend_index: float) -> float:
    context = dict(market_context or {})
    regime = str(context.get("market_regime", "")).lower()
    if regime in {"event_compression", "market_wide_sellout"}:
        return 1.0
    if retained_pace_index > 1.2 or pickup_trend_index > 1.2:
        return 1.0
    return 0.0


def _event_text_from_calendar(event: Dict[str, Any]) -> str:
    parts = [
        event.get("event_name", ""),
        event.get("event_category", ""),
        event.get("area_or_venue", ""),
        event.get("notes", ""),
    ]
    attendance = _safe_float(event.get("expected_attendance"), 0.0)
    if attendance:
        parts.append(f"{int(attendance)} attendees")
    return " ".join(str(part) for part in parts if str(part).strip())


def _is_generic_calendar_request(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    generic_requests = {
        "",
        "event",
        "local event",
        "local intel",
        "local intel event",
        "local intel overlay",
        "local intel event overlay",
        "calendar event",
        "seeded event",
        "seeded local intel",
        "seeded local-intel overlay",
    }
    return cleaned in generic_requests


def _mentions_calendar_event(text: str, events: List[Dict[str, Any]]) -> bool:
    normalized = (text or "").lower()
    return any(str(event.get("event_name", "")).lower() in normalized for event in events)


def _cap_occupancy_shock(
    raw_shock: float,
    current_occ: float,
    forecast_occ: float,
    guardrails: List[str],
) -> tuple[float, float, str]:
    available_shock = max(0.0, 1.0 - max(current_occ, forecast_occ))
    if raw_shock > available_shock:
        guardrails.append("Occupancy shock was clamped so event-adjusted demand cannot exceed 100%.")
        reason = (
            "Demand impact was capped because current or forecast occupancy leaves only "
            f"{available_shock * 100:.1f}% occupancy room."
        )
        return available_shock, available_shock, reason
    return raw_shock, available_shock, ""


def _combine_event_results(
    calendar_result: Dict[str, Any],
    manual_result: Dict[str, Any],
    *,
    current_occ: float,
    forecast_occ: float,
) -> Dict[str, Any]:
    baseline_occ = max(current_occ, forecast_occ)
    available_shock = max(0.0, 1.0 - baseline_occ)
    calendar_shock = _safe_float(
        calendar_result.get("raw_occupancy_shock", calendar_result.get("suggested_shock")),
        0.0,
    )
    manual_shock = max(
        0.0,
        _safe_float(
            manual_result.get("raw_occupancy_shock", manual_result.get("suggested_shock")),
            0.0,
        ),
    )
    raw_cluster_shock = calendar_shock + (0.65 * manual_shock)
    suggested_shock = min(MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, raw_cluster_shock, available_shock)
    demand_clamp_reason = ""

    calendar_headroom = _safe_float(calendar_result.get("adr_headroom"), 0.0)
    manual_headroom = max(0.0, _safe_float(manual_result.get("adr_headroom"), 0.0))
    adr_headroom = min(
        MAX_ADR_HEADROOM,
        max(calendar_headroom, manual_headroom) + (0.50 * min(calendar_headroom, manual_headroom)),
    )

    guardrails = (
        list(calendar_result.get("guardrails_applied", []))
        + list(manual_result.get("guardrails_applied", []))
        + [
            "Calendar and manager-entered local intel were combined as an event cluster.",
            "Secondary manual-event impact is discounted to avoid double-counting overlapping local demand.",
        ]
    )
    if raw_cluster_shock > suggested_shock:
        guardrails.append("Cluster demand impact was capped by occupancy and event-impact guardrails.")
        if suggested_shock == available_shock:
            demand_clamp_reason = (
                "Cluster demand impact was capped because current or forecast occupancy leaves only "
                f"{available_shock * 100:.1f}% occupancy room."
            )
        else:
            demand_clamp_reason = "Cluster demand impact was capped by the maximum local-intel event-impact guardrail."

    calendar_evidence = list(calendar_result.get("evidence", []))
    manual_evidence = list(manual_result.get("evidence", []))
    confidence = "High" if "High" in {calendar_result.get("confidence"), manual_result.get("confidence")} else "Medium"
    intensity = min(
        1.0,
        max(
            _safe_float(calendar_result.get("event_intensity_score"), 0.0),
            _safe_float(manual_result.get("event_intensity_score"), 0.0),
        )
        + 0.15,
    )
    rationale = (
        "Known calendar demand and manager-entered local intel were combined as a Lisbon city-hotel event cluster. "
        "The baseline forecast is unchanged; the combined demand and ADR-headroom estimate only affects a manager-approved scenario."
    )
    return _build_result(
        classification="Local Event Cluster",
        suggested_shock=suggested_shock,
        adr_headroom=adr_headroom,
        confidence=confidence,
        rationale=rationale,
        guardrails=guardrails,
        apply_allowed=True,
        event_intensity_score=intensity,
        evidence=calendar_evidence + manual_evidence,
        source="seeded_calendar_plus_manual_intel",
        hotel_location_profile=HOTEL_LOCATION_PROFILE,
        calendar_events=list(calendar_result.get("calendar_events", [])),
        raw_occupancy_shock=raw_cluster_shock,
        available_occupancy_shock=available_shock,
        demand_clamp_reason=demand_clamp_reason,
        proximity_bucket="event_cluster",
        proximity_source="calendar_plus_manual",
        proximity_factor=1.0,
        proximity_evidence=[
            calendar_result.get("proximity_bucket", ""),
            manual_result.get("proximity_bucket", ""),
        ],
    )


def _calendar_result(
    events: List[Dict[str, Any]],
    *,
    current_occ: float,
    forecast_occ: float,
    retained_pace_index: float,
    pickup_trend_index: float,
    market_context=None,
) -> Dict[str, Any]:
    profiles = []
    evidence = []
    guardrails = [
        "Local intel is an external September 2017 Lisbon overlay, not a field learned from the source booking dataset.",
        "The source hotel is anonymized, so scoring uses venue relevance buckets rather than exact hotel distance.",
        "Local intel is an estimate and is never applied automatically.",
    ]
    for event in events:
        attendance = _safe_float(event.get("expected_attendance"), 0.0)
        profile = _category_profile(event.get("event_category", ""), attendance)
        factor = _proximity_factor(event.get("proximity_bucket", ""))
        shock = profile["base_shock"] * factor
        headroom = profile["base_headroom"] * factor
        profiles.append(
            {
                **profile,
                "shock": shock,
                "headroom": headroom,
                "event": event,
            }
        )
        attendance_text = f", attendance about {int(attendance):,}" if attendance else ""
        evidence.append(
            f"{event.get('event_name')} ({event.get('area_or_venue')}{attendance_text}; source: {event.get('source_quality')})"
        )

    strongest = max(profiles, key=lambda row: (row["shock"], row["headroom"])) if profiles else None
    if not strongest:
        return _build_result(
            classification="Irrelevant",
            suggested_shock=0.0,
            adr_headroom=0.0,
            confidence="Low",
            rationale="No local intel was supplied.",
            guardrails=["No local intel impact was estimated."],
            apply_allowed=False,
        )

    market_boost = _market_proof_boost(market_context, retained_pace_index, pickup_trend_index)
    suggested_shock = strongest["shock"]
    adr_headroom = strongest["headroom"]
    if market_boost:
        if strongest["classification"] == "Major Sports Event":
            suggested_shock = max(suggested_shock, 0.18)
            adr_headroom = max(adr_headroom, 0.20)
        else:
            suggested_shock = min(MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, suggested_shock + 0.02)
            adr_headroom = min(MAX_ADR_HEADROOM, adr_headroom + 0.03)
        guardrails.append("Market proof or pickup strength allowed the upper event-impact tier.")

    raw_suggested_shock = suggested_shock
    suggested_shock, available_shock, demand_clamp_reason = _cap_occupancy_shock(
        raw_suggested_shock,
        current_occ,
        forecast_occ,
        guardrails,
    )

    event_count = len(events)
    classification = strongest["classification"]
    if event_count > 1 and classification != "Major Sports Event":
        classification = "Local Event Cluster"
    event_names = ", ".join(event.get("event_name", "local event") for event in events[:2])
    if event_count > 2:
        event_names += f" and {event_count - 2} more"
    rationale = (
        f"{event_names} is treated as a Lisbon city-hotel local-intel overlay for this September 2017 stay date. "
        f"The baseline forecast is unchanged; this estimate only affects a manager-approved scenario."
    )
    return _build_result(
        classification=classification,
        suggested_shock=suggested_shock,
        adr_headroom=adr_headroom,
        confidence=strongest["confidence"],
        rationale=rationale,
        guardrails=guardrails,
        apply_allowed=True,
        event_intensity_score=strongest["intensity"],
        evidence=evidence,
        source="seeded_lisbon_calendar",
        hotel_location_profile=HOTEL_LOCATION_PROFILE,
        calendar_events=events,
        raw_occupancy_shock=raw_suggested_shock,
        available_occupancy_shock=available_shock,
        demand_clamp_reason=demand_clamp_reason,
        proximity_bucket=strongest["event"].get("proximity_bucket", "distant_or_uncertain"),
        proximity_source="seeded_calendar",
        proximity_factor=_proximity_factor(strongest["event"].get("proximity_bucket", "")),
        proximity_evidence=[strongest["event"].get("area_or_venue", "")],
    )


def _build_result(
    classification: str,
    suggested_shock: float,
    confidence: str,
    rationale: str,
    guardrails,
    apply_allowed: bool,
    adr_headroom: float = 0.0,
    event_intensity_score: float | None = None,
    evidence: List[str] | None = None,
    source: str = "deterministic_text_parser",
    hotel_location_profile: str = HOTEL_LOCATION_PROFILE,
    calendar_events: List[Dict[str, Any]] | None = None,
    raw_occupancy_shock: float | None = None,
    available_occupancy_shock: float | None = None,
    demand_clamp_reason: str = "",
    proximity_bucket: str = "",
    proximity_source: str = "",
    proximity_factor: float | None = None,
    proximity_evidence: List[str] | None = None,
) -> Dict[str, Any]:
    capped = max(MIN_DEFAULT_SHOCK, min(MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, suggested_shock))
    raw_occupancy_shock = suggested_shock if raw_occupancy_shock is None else float(raw_occupancy_shock or 0.0)
    raw_occupancy_shock = max(MIN_DEFAULT_SHOCK, min(MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, raw_occupancy_shock))
    available_occupancy_shock = (
        capped if available_occupancy_shock is None else max(0.0, float(available_occupancy_shock or 0.0))
    )
    adr_headroom = max(0.0, min(MAX_ADR_HEADROOM, float(adr_headroom or 0.0)))
    guardrail_list = list(guardrails)
    if capped != suggested_shock:
        guardrail_list.append("Suggested local intel impact was capped by estimator guardrails.")

    if confidence == "Low" and (capped != 0 or adr_headroom != 0):
        capped = 0.0
        raw_occupancy_shock = 0.0
        adr_headroom = 0.0
        apply_allowed = False
        guardrail_list.append("Low-confidence local intel cannot be applied to baseline.")

    if capped == 0 and adr_headroom == 0:
        apply_allowed = False

    if event_intensity_score is None:
        event_intensity_score = min(1.0, max(abs(capped) / MAX_HIGH_CONFIDENCE_DEMAND_SHOCK, adr_headroom / MAX_ADR_HEADROOM))

    demand_was_clamped = round(raw_occupancy_shock, 4) != round(capped, 4)

    return {
        "classification": classification,
        "event_intensity_score": round(event_intensity_score, 4),
        "occupancy_shock": round(capped, 4),
        "occupancy_shock_pct": round(capped * 100, 1),
        "applied_occupancy_shock": round(capped, 4),
        "applied_occupancy_shock_pct": round(capped * 100, 1),
        "raw_occupancy_shock": round(raw_occupancy_shock, 4),
        "raw_occupancy_shock_pct": round(raw_occupancy_shock * 100, 1),
        "available_occupancy_shock": round(available_occupancy_shock, 4),
        "available_occupancy_shock_pct": round(available_occupancy_shock * 100, 1),
        "demand_was_clamped": demand_was_clamped,
        "demand_clamp_reason": demand_clamp_reason if demand_was_clamped else "",
        "suggested_shock_pct": round(capped * 100, 1),
        "suggested_shock": round(capped, 4),
        "adr_headroom": round(adr_headroom, 4),
        "adr_headroom_pct": round(adr_headroom * 100, 1),
        "confidence": confidence,
        "rationale": rationale,
        "manager_rationale": rationale,
        "evidence": evidence or [],
        "source": source,
        "hotel_location_profile": hotel_location_profile,
        "calendar_events": calendar_events or [],
        "proximity_bucket": proximity_bucket,
        "proximity_label": PROXIMITY_LABELS.get(proximity_bucket, proximity_bucket),
        "proximity_source": proximity_source,
        "proximity_factor": round(float(proximity_factor if proximity_factor is not None else 1.0), 2),
        "proximity_evidence": [item for item in (proximity_evidence or []) if item],
        "guardrails_applied": guardrail_list,
        "apply_allowed": apply_allowed,
    }


def estimate_local_intel_impact(
    text: str,
    current_occ: float,
    forecast_occ: float,
    booking_velocity: float = 1.0,
    retained_pace_index: float | None = None,
    pickup_trend_index: float | None = None,
    target_date=None,
    market_context=None,
    manual_proximity_bucket: str | None = None,
) -> Dict[str, Any]:
    """Estimate event demand and ADR headroom without letting AI own ADR.

    The estimator returns legacy `suggested_shock` keys plus a richer production
    MVP profile. Any non-zero impact is still advisory until explicitly applied.
    """
    normalized = (text or "").strip().lower()
    current_occ = max(0.0, min(1.0, float(current_occ or 0.0)))
    forecast_occ = max(0.0, min(1.0, float(forecast_occ or 0.0)))
    booking_velocity = float(booking_velocity or 1.0)
    retained_pace_index = float(retained_pace_index if retained_pace_index is not None else booking_velocity)
    pickup_trend_index = float(pickup_trend_index if pickup_trend_index is not None else booking_velocity)

    calendar_events = local_intel_events_for_date(target_date) if target_date else []
    if calendar_events:
        calendar_result = _calendar_result(
            calendar_events,
            current_occ=current_occ,
            forecast_occ=forecast_occ,
            retained_pace_index=retained_pace_index,
            pickup_trend_index=pickup_trend_index,
            market_context=market_context,
        )
        if _is_generic_calendar_request(normalized) or _mentions_calendar_event(normalized, calendar_events):
            return calendar_result
        manual_result = estimate_local_intel_impact(
            text,
            current_occ=current_occ,
            forecast_occ=forecast_occ,
            booking_velocity=booking_velocity,
            retained_pace_index=retained_pace_index,
            pickup_trend_index=pickup_trend_index,
            target_date=None,
            market_context=market_context,
            manual_proximity_bucket=manual_proximity_bucket,
        )
        if manual_result.get("apply_allowed") and (
            _safe_float(manual_result.get("suggested_shock"), 0.0) != 0
            or _safe_float(manual_result.get("adr_headroom"), 0.0) != 0
        ):
            return _combine_event_results(
                calendar_result,
                manual_result,
                current_occ=current_occ,
                forecast_occ=forecast_occ,
            )
        return calendar_result

    if not normalized:
        return _build_result(
            classification="Irrelevant",
            suggested_shock=0.0,
            adr_headroom=0.0,
            confidence="Low",
            rationale="No local intel was supplied.",
            guardrails=["No local intel impact was estimated."],
            apply_allowed=False,
        )

    demand_terms = [
        "wedding", "marriage", "banquet", "concert", "conference", "convention",
        "expo", "summit", "festival", "tournament", "match", "event nearby",
        "fifa", "world cup", "cup final", "final", "semi final", "semifinal",
        "championship", "football", "cricket", "ipl", "olympics", "grand prix",
        "race", "derby", "benfica", "sporting", "champions league",
    ]
    sellout_terms = ["sold out", "sold-out", "nearby hotels full", "hotels full", "all hotels full", "city sold out"]
    disruption_terms = [
        "traffic", "traffic jam", "road closure", "blocked road", "strike",
        "protest", "weather", "storm", "flood", "airport shutdown", "cancelled flight",
        "canceled flight", "entry of the city",
    ]
    stranded_terms = ["stranded", "overnight stay", "last minute rooms", "walk-in", "walk ins"]
    competitor_terms = ["competitor", "comp set", "nearby hotel", "other hotels", "market rate"]

    guardrails = ["Local intel is an estimate and is never applied automatically."]
    proximity = infer_local_intel_proximity(normalized, manual_proximity_bucket)
    proximity_factor = _safe_float(proximity.get("factor"), 1.0)
    if proximity["source"] == "manager_override":
        guardrails.append(
            f"Manual proximity override set to {proximity['label']} ({proximity_factor:.2f}x impact factor)."
        )
    elif proximity["source"] == "text_inferred":
        matched = ", ".join(proximity.get("matched_terms", []))
        guardrails.append(
            f"Manual proximity inferred as {proximity['label']} from text ({proximity_factor:.2f}x impact factor"
            f"{': ' + matched if matched else ''})."
        )

    has_demand_signal = _contains_any(normalized, demand_terms)
    has_sellout_signal = _contains_any(normalized, sellout_terms)
    is_major_sports_event = _is_major_sports_event(normalized)
    market_boost = _market_proof_boost(market_context, retained_pace_index, pickup_trend_index)

    def build_manual_result(
        *,
        classification: str,
        base_shock: float,
        base_headroom: float,
        confidence: str,
        rationale: str,
        extra_guardrails: List[str],
        apply_allowed: bool,
        event_intensity_score: float | None = None,
    ) -> Dict[str, Any]:
        adjusted_shock = base_shock * proximity_factor if base_shock > 0 else base_shock
        adjusted_headroom = min(MAX_ADR_HEADROOM, base_headroom * proximity_factor) if base_headroom > 0 else base_headroom
        if adjusted_shock > 0:
            capped_shock, available_shock, clamp_reason = _cap_occupancy_shock(
                adjusted_shock,
                current_occ,
                forecast_occ,
                guardrails,
            )
        else:
            capped_shock = adjusted_shock
            available_shock = max(0.0, 1.0 - max(current_occ, forecast_occ))
            clamp_reason = ""
        return _build_result(
            classification=classification,
            suggested_shock=capped_shock,
            adr_headroom=adjusted_headroom,
            confidence=confidence,
            rationale=rationale,
            guardrails=guardrails + extra_guardrails,
            apply_allowed=apply_allowed,
            event_intensity_score=event_intensity_score,
            evidence=[text.strip()],
            raw_occupancy_shock=adjusted_shock,
            available_occupancy_shock=available_shock,
            demand_clamp_reason=clamp_reason,
            proximity_bucket=proximity["bucket"],
            proximity_source=proximity["source"],
            proximity_factor=proximity_factor,
            proximity_evidence=proximity.get("matched_terms", []),
        )

    if _contains_any(normalized, disruption_terms):
        if _contains_any(normalized, stranded_terms):
            return build_manual_result(
                classification="Disruption With Room Demand",
                base_shock=0.05,
                base_headroom=0.03,
                confidence="Medium",
                rationale="The disruption may create short-notice room demand from stranded travelers.",
                extra_guardrails=["Disruption-driven demand is capped unless externally validated."],
                apply_allowed=True,
            )

        if "airport shutdown" in normalized or "cancelled flight" in normalized or "canceled flight" in normalized:
            return build_manual_result(
                classification="Operational Disruption / Ambiguous Demand",
                base_shock=-0.03,
                base_headroom=0.0,
                confidence="Medium",
                rationale="Airport disruption may reduce arrivals unless there is evidence of stranded-room demand.",
                extra_guardrails=["Disruption signals default conservative and require user approval."],
                apply_allowed=True,
            )

        return build_manual_result(
            classification="Operational Disruption / Ambiguous Demand",
            base_shock=0.0,
            base_headroom=0.0,
            confidence="Low",
            rationale="Traffic or access disruption is ambiguous; it may delay arrivals without increasing hotel demand.",
            extra_guardrails=["Traffic, weather, protests, and road closures default to context-only unless room demand is explicit."],
            apply_allowed=False,
        )

    if has_demand_signal:
        if has_sellout_signal:
            shock = MAX_HIGH_CONFIDENCE_DEMAND_SHOCK if market_boost else MAX_DEFAULT_SHOCK + 0.05
            headroom = 0.20 if market_boost else 0.15
            return build_manual_result(
                classification="Event",
                base_shock=shock,
                base_headroom=headroom,
                confidence="High",
                rationale="The local event includes strong sell-out language, suggesting high compression demand.",
                extra_guardrails=["High-confidence demand event can affect demand and ADR headroom only after approval."],
                apply_allowed=True,
                event_intensity_score=0.85,
            )

        if is_major_sports_event:
            suggested = 0.05
            headroom = 0.08
            if forecast_occ > 0.90 or retained_pace_index > 1.2 or pickup_trend_index > 1.2:
                suggested = 0.12
                headroom = 0.15
            elif forecast_occ > 0.80 or retained_pace_index > 1.1 or pickup_trend_index > 1.1:
                suggested = 0.08
                headroom = 0.10
            if market_boost:
                suggested = max(suggested, 0.18)
                headroom = max(headroom, 0.20)
            return build_manual_result(
                classification="Major Sports Event",
                base_shock=suggested,
                base_headroom=headroom,
                confidence="Medium",
                rationale="Major sports demand can create local compression, but impact is not applied unless approved.",
                extra_guardrails=["Sports-event impact uses occupancy shock plus bounded ADR headroom."],
                apply_allowed=True,
                event_intensity_score=0.75,
            )

        if _has_numbered_group(normalized) or _contains_any(normalized, ["wedding", "conference", "convention", "expo", "summit"]):
            suggested = 0.03 if _contains_any(normalized, ["conference", "convention", "expo", "summit"]) else 0.08
            headroom = 0.04 if suggested <= 0.03 else 0.06
            if forecast_occ > 0.90 or retained_pace_index > 1.2 or pickup_trend_index > 1.2:
                suggested = min(0.10, suggested + 0.02)
                headroom = min(0.08, headroom + 0.02)
            return build_manual_result(
                classification="Business Event" if suggested <= 0.05 else "Event",
                base_shock=suggested,
                base_headroom=headroom,
                confidence="Medium",
                rationale="The local intel describes a plausible demand-generating event.",
                extra_guardrails=["Standard event impact is bounded unless stronger market proof is supplied."],
                apply_allowed=True,
                event_intensity_score=0.45 if suggested <= 0.05 else 0.55,
            )

        return build_manual_result(
            classification="Event",
            base_shock=0.05,
            base_headroom=0.04,
            confidence="Medium",
            rationale="The local intel may generate incremental lodging demand, but details are limited.",
            extra_guardrails=["Limited event details keep the suggested impact conservative."],
            apply_allowed=True,
        )

    if _contains_any(normalized, competitor_terms):
        return build_manual_result(
            classification="Competitor Signal",
            base_shock=0.0,
            base_headroom=0.0,
            confidence="Low",
            rationale="The text appears to describe competitor context rather than direct demand impact.",
            extra_guardrails=["Competitor signals should affect benchmarking, not occupancy shock."],
            apply_allowed=False,
        )

    return build_manual_result(
        classification="Ambiguous",
        base_shock=0.0,
        base_headroom=0.0,
        confidence="Low",
        rationale="The local intel does not clearly indicate a measurable demand change.",
        extra_guardrails=["Ambiguous intel is treated as context only."],
        apply_allowed=False,
    )
