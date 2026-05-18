import json
from typing import Any, Dict, Iterable, List

import pandas as pd

from config import BASE_CAPACITY, CHAT_MODEL
from pricing_agent import _get_client, _resolve_api_key
from pricing_engine import calculate_recommended_price


def _safe_float(value, fallback=0.0) -> float:
    try:
        number = float(value)
        if pd.notna(number):
            return number
    except (TypeError, ValueError):
        pass
    return fallback


def _nearest_candidate_for_price(breakdown: Dict[str, Any], price: float) -> Dict[str, Any]:
    candidates = breakdown.get("optimizer_candidates", [])
    if not candidates:
        expected_rooms = _safe_float(breakdown.get("expected_rooms"))
        return {
            "price": price,
            "expected_rooms": expected_rooms,
            "expected_revenue": round(price * expected_rooms, 2),
        }
    return min(
        candidates,
        key=lambda row: (
            abs(_safe_float(row.get("price")) - price),
            _safe_float(row.get("price")),
        ),
    )


def _manual_approval_required(
    breakdown: Dict[str, Any],
    final_price: float,
    review_flags: Iterable[str],
) -> bool:
    reference_price = _safe_float(breakdown.get("reference_price"))
    absolute_delta = final_price - reference_price
    pct_delta = (absolute_delta / reference_price) * 100 if reference_price else 0.0
    return (
        abs(pct_delta) > 20
        or abs(absolute_delta) > 30
        or any("review" in str(flag).lower() for flag in review_flags)
    )


def _top_reasons(breakdown: Dict[str, Any], revenue_upside: float, review_flags: List[str]) -> List[str]:
    reasons = []
    if revenue_upside > 0:
        reasons.append(f"${revenue_upside:,.0f} upside versus booked ADR.")
    if breakdown.get("sold_out"):
        reasons.append("Raw OTB is sold out; protect the scarce remaining inventory.")
    elif _safe_float(breakdown.get("compression_score")) >= 0.60:
        reasons.append("Demand compression is strong.")
    if _safe_float(breakdown.get("pickup_trend_index"), 1.0) >= 1.20:
        reasons.append("Recent pickup is accelerating.")
    reasons.extend(str(flag) for flag in review_flags[:2])
    return reasons[:3]


def build_opportunity_records(forecast_df: pd.DataFrame, live_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one deterministic opportunity record per forecast date."""
    records = []
    for row in forecast_df.sort_values("Date").itertuples(index=False):
        date = pd.to_datetime(row.Date)
        date_key = date.strftime("%Y-%m-%d")
        live_entry = live_data.get(date_key, {})
        total_rooms = max(_safe_float(live_entry.get("total_rooms"), BASE_CAPACITY), 1.0)
        raw_otb_occupancy = _safe_float(
            live_entry.get("raw_otb_occupancy"),
            _safe_float(live_entry.get("current_otb"), 0.0) / total_rooms,
        )
        adjusted_otb_occupancy = _safe_float(
            live_entry.get("adjusted_otb_occupancy"),
            raw_otb_occupancy,
        )
        forecasted_occupancy = _safe_float(getattr(row, "Forecasted_Occupancy", 0.0))
        occupancy_for_pricing = max(adjusted_otb_occupancy, forecasted_occupancy)
        competitor_price = _safe_float(live_entry.get("competitor_price"), _safe_float(getattr(row, "Competitor_Rate", 0.0), 120.0))
        market_context = {
            "comp_low": live_entry.get("comp_low"),
            "comp_median": live_entry.get("comp_median"),
            "comp_high": live_entry.get("comp_high"),
            "sample_size": live_entry.get("sample_size"),
            "source_quality": live_entry.get("source_quality"),
            "market_regime": live_entry.get("market_regime"),
            "market_as_of_timestamp": live_entry.get("market_as_of_timestamp"),
        }
        final_price, _, breakdown = calculate_recommended_price(
            occupancy=occupancy_for_pricing,
            day_name=date.strftime("%A"),
            target_date=date_key,
            competitor_price=competitor_price,
            market_context=market_context,
            return_breakdown=True,
            booking_velocity=live_entry.get("booking_velocity", 1.0),
            gross_pace_index=live_entry.get("gross_pace_index"),
            retained_pace_index=live_entry.get("retained_pace_index"),
            pickup_trend_index=live_entry.get("pickup_trend_index"),
            pricing_pace_index=live_entry.get("pricing_pace_index"),
            raw_otb_occupancy=raw_otb_occupancy,
            adjusted_otb_occupancy=adjusted_otb_occupancy,
            expected_cancellations=live_entry.get("expected_cancellations", 0.0),
        )
        reference_price = _safe_float(breakdown.get("reference_price"))
        booked_adr = _safe_float(live_entry.get("booked_adr"), reference_price)
        reference_candidate = _nearest_candidate_for_price(breakdown, reference_price)
        booked_adr_candidate = _nearest_candidate_for_price(breakdown, booked_adr)
        expected_revenue = _safe_float(breakdown.get("expected_revenue"))
        reference_revenue_proxy = _safe_float(reference_candidate.get("expected_revenue"))
        booked_adr_revenue_proxy = _safe_float(booked_adr_candidate.get("expected_revenue"))
        revenue_upside = max(0.0, round(expected_revenue - booked_adr_revenue_proxy, 2))
        review_flags = list(breakdown.get("review_flags", []))
        manual_approval_required = _manual_approval_required(breakdown, final_price, review_flags)
        if breakdown.get("sold_out") and breakdown.get("material_retention_gap"):
            review_flags = review_flags + [
                "Sold-out OTB and lower retained occupancy need a remaining-room strategy review."
            ]
            manual_approval_required = True
        review_status = "Review needed" if review_flags or manual_approval_required else "No review"
        records.append(
            {
                "date": date_key,
                "recommended_adr": round(final_price, 2),
                "reference_adr": round(reference_price, 2),
                "booked_adr": round(booked_adr, 2),
                "expected_revenue": round(expected_revenue, 2),
                "reference_revenue_proxy": round(reference_revenue_proxy, 2),
                "booked_adr_revenue_proxy": round(booked_adr_revenue_proxy, 2),
                "revenue_upside": round(revenue_upside, 2),
                "review_status": review_status,
                "review_flags": review_flags,
                "top_reasons": _top_reasons(breakdown, revenue_upside, review_flags),
                "raw_otb_occupancy": round(raw_otb_occupancy, 4),
                "adjusted_otb_occupancy": round(adjusted_otb_occupancy, 4),
                "forecasted_occupancy": round(forecasted_occupancy, 4),
                "pickup_trend_index": round(_safe_float(live_entry.get("pickup_trend_index"), 1.0), 4),
                "competitor_median": round(_safe_float(breakdown.get("competitor_price"), competitor_price), 2),
                "manual_approval_required": manual_approval_required,
                "sold_out": bool(breakdown.get("sold_out")),
                "material_retention_gap": bool(breakdown.get("material_retention_gap")),
                "pricing_breakdown": breakdown,
            }
        )
    return records


def rank_top_opportunities(records: Iterable[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (item["revenue_upside"], item["expected_revenue"], item["date"]),
        reverse=True,
    )[:limit]


def rank_top_risks(records: Iterable[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    risk_records = [record for record in records if record["review_status"] == "Review needed"]
    return sorted(
        risk_records,
        key=lambda item: (
            item["manual_approval_required"],
            item["sold_out"] and item["material_retention_gap"],
            len(item["review_flags"]),
            item["revenue_upside"],
            item["date"],
        ),
        reverse=True,
    )[:limit]


def build_summary_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    return {
        "dates_evaluated": len(rows),
        "dates_with_upside": sum(1 for row in rows if row["revenue_upside"] > 0),
        "dates_needing_review": sum(1 for row in rows if row["review_status"] == "Review needed"),
        "sold_out_dates": sum(1 for row in rows if row["sold_out"]),
        "total_revenue_upside": round(sum(row["revenue_upside"] for row in rows), 2),
    }


def build_briefing_payload(records: Iterable[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    rows = list(records)
    return {
        "top_opportunities": [
            {
                "date": row["date"],
                "recommended_adr": row["recommended_adr"],
                "booked_adr": row["booked_adr"],
                "revenue_upside": row["revenue_upside"],
                "top_reasons": row["top_reasons"],
            }
            for row in rank_top_opportunities(rows, limit=limit)
        ],
        "top_risks": [
            {
                "date": row["date"],
                "recommended_adr": row["recommended_adr"],
                "review_flags": row["review_flags"][:2],
            }
            for row in rank_top_risks(rows, limit=limit)
        ],
        "summary_metrics": build_summary_metrics(rows),
    }


def deterministic_executive_briefing(payload: Dict[str, Any]) -> str:
    opportunities = payload.get("top_opportunities", [])
    risks = payload.get("top_risks", [])
    metrics = payload.get("summary_metrics", {})
    if not opportunities:
        opportunity_sentence = "No material revenue upside stands out across the next 30 days."
    else:
        lead = opportunities[0]
        opportunity_sentence = (
            f"Focus first on {lead['date']}: the current recommendation shows about "
            f"${lead['revenue_upside']:,.0f} of upside versus booked ADR."
        )
    if not risks:
        risk_sentence = "No dates currently require special review before publishing."
    else:
        lead_risk = risks[0]
        risk_sentence = (
            f"Review {lead_risk['date']} before publishing because "
            f"{lead_risk['review_flags'][0].rstrip('.') if lead_risk['review_flags'] else 'the date carries elevated pricing risk'}."
        )
    summary_sentence = (
        f"Across the next {metrics.get('dates_evaluated', 0)} days, "
        f"{metrics.get('dates_with_upside', 0)} dates show upside and "
        f"{metrics.get('dates_needing_review', 0)} need review."
    )
    return f"{opportunity_sentence} {risk_sentence} {summary_sentence}"


def _briefing_needs_fallback(briefing: str, payload: Dict[str, Any]) -> bool:
    normalized = (briefing or "").lower()
    forbidden_phrases = [
        "raise rates",
        "lower rates",
        "replace the adr",
        "change adr",
        "current bookings",
    ]
    if not normalized.strip():
        return True
    if any(phrase in normalized for phrase in forbidden_phrases):
        return True
    if payload.get("top_opportunities") and "$" not in briefing:
        return True
    return False


def generate_executive_briefing(payload: Dict[str, Any]) -> str:
    """Return a short manager-facing briefing; AI may summarize, never change ADR."""
    fallback = deterministic_executive_briefing(payload)
    if not _resolve_api_key():
        return fallback

    prompt = f"""
You are writing the morning executive briefing for a single-property hotel manager.
Use only the supplied structured data. Do not invent prices, do not change ADR, and do not recommend replacement rates.
Write 2-3 concise sentences covering:
1. the most important revenue opportunity,
2. the most important review risk,
3. the overall 30-day picture.
Keep the tone crisp, commercial, and manager-friendly. Avoid technical jargon.
Refer to the already-calculated recommendations; do not say "raise rates" or "lower rates."
Use the "$" symbol whenever you mention money.
When describing upside, say "versus booked ADR," not "versus bookings."

Structured data:
{json.dumps(payload, ensure_ascii=False)}

Return valid JSON only:
{{"executive_briefing": "brief text"}}
"""
    try:
        response = _get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hotel revenue strategy reviewer. "
                        "The optimizer owns ADR; you summarize only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = json.loads(response.choices[0].message.content)
        briefing = str(content.get("executive_briefing", "")).strip()
        return fallback if _briefing_needs_fallback(briefing, payload) else briefing
    except Exception:
        return fallback
