import pandas as pd
import numpy as np

from config import (
    DATA_PATH,
    FORECAST_OUTPUT_PATH,
    MIN_PRICE,
    MAX_PRICE,
    BASE_PRICE,
    BASE_CAPACITY,
    PRICE_STEP,
    BASE_RATE_LOOKBACK_DAYS,
    DYNAMIC_BASE_DOW_WEIGHT,
    DYNAMIC_FLOOR_COMP_LOW_FACTOR,
    DYNAMIC_CEILING_BASE_MULTIPLIER,
)

SOLD_OUT_THRESHOLD = 0.9999
MATERIAL_OCCUPANCY_GAP = 0.05


def _round_money(value):
    return round(float(value), 2)


def _add_component(components, label, before, after, explanation):
    before = float(before)
    after = float(after)
    components.append(
        {
            "driver": label,
            "price_before": _round_money(before),
            "adjustment": _round_money(after - before),
            "price_after": _round_money(after),
            "explanation": explanation,
        }
    )


def _finite_float(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def normalize_market_context(market_context=None, competitor_price=None):
    """Normalize richer comp-set inputs while preserving legacy single-rate callers."""
    context = dict(market_context or {})
    legacy_price = _finite_float(competitor_price)
    median = _finite_float(context.get("comp_median"))
    if median is None:
        median = legacy_price

    if median is None:
        return {
            "comp_low": None,
            "comp_median": None,
            "comp_high": None,
            "sample_size": 0,
            "source_quality": context.get("source_quality", "unavailable"),
            "market_regime": context.get("market_regime", "unavailable"),
            "as_of_timestamp": context.get("as_of_timestamp") or context.get("market_as_of_timestamp"),
        }

    low = median if _finite_float(context.get("comp_low")) is None else float(context["comp_low"])
    high = median if _finite_float(context.get("comp_high")) is None else float(context["comp_high"])
    low, median, high = sorted([float(low), float(median), float(high)])
    return {
        "comp_low": _round_money(low),
        "comp_median": _round_money(median),
        "comp_high": _round_money(high),
        "sample_size": int(_finite_float(context.get("sample_size")) or (1 if legacy_price is not None else 0)),
        "source_quality": context.get("source_quality", "legacy_single_rate" if legacy_price is not None else "unavailable"),
        "market_regime": context.get("market_regime", "legacy_single_rate" if legacy_price is not None else "unavailable"),
        "as_of_timestamp": context.get("as_of_timestamp") or context.get("market_as_of_timestamp"),
    }


def _candidate_prices(min_price=MIN_PRICE, max_price=MAX_PRICE):
    start = int(np.ceil(min_price / PRICE_STEP) * PRICE_STEP)
    end = int(np.floor(max_price / PRICE_STEP) * PRICE_STEP)
    if end < start:
        start = end = int(round(float(min_price) / PRICE_STEP) * PRICE_STEP)
    return [float(price) for price in range(start, end + PRICE_STEP, PRICE_STEP)]


def _is_sold_out(raw_otb_occupancy):
    return raw_otb_occupancy is not None and raw_otb_occupancy >= SOLD_OUT_THRESHOLD


def _has_material_retention_gap(raw_otb_occupancy, adjusted_otb_occupancy):
    return (
        raw_otb_occupancy is not None
        and adjusted_otb_occupancy is not None
        and (raw_otb_occupancy - adjusted_otb_occupancy) >= MATERIAL_OCCUPANCY_GAP
    )


def _resolve_pace_signals(
    booking_velocity=1.0,
    gross_pace_index=None,
    retained_pace_index=None,
    pickup_trend_index=None,
    pricing_pace_index=None,
):
    legacy_velocity = max(0.1, float(booking_velocity or 1.0))
    gross = max(0.1, float(gross_pace_index if gross_pace_index is not None else legacy_velocity))
    retained = max(0.1, float(retained_pace_index if retained_pace_index is not None else gross))
    pickup = max(0.1, float(pickup_trend_index if pickup_trend_index is not None else gross))
    pricing = (
        max(0.1, float(pricing_pace_index))
        if pricing_pace_index is not None
        else legacy_velocity
        if all(value is None for value in [gross_pace_index, retained_pace_index, pickup_trend_index])
        else max(0.1, min(2.0, (0.20 * gross) + (0.50 * retained) + (0.30 * pickup)))
    )
    return {
        "booking_velocity": gross,
        "gross_pace_index": gross,
        "retained_pace_index": retained,
        "pickup_trend_index": pickup,
        "pricing_pace_index": pricing,
    }


def _dynamic_elasticity(occupancy, pricing_pace_index, local_intel_shock):
    """Transparent elasticity proxy until a learned demand model replaces it."""
    elasticity = 1.35
    if occupancy >= 0.90:
        elasticity -= 0.35
    elif occupancy >= 0.80:
        elasticity -= 0.20
    elif occupancy < 0.55:
        elasticity += 0.30

    if pricing_pace_index >= 1.25:
        elasticity -= 0.20
    elif pricing_pace_index <= 0.80:
        elasticity += 0.25

    if local_intel_shock > 0:
        elasticity -= min(0.15, local_intel_shock)
    elif local_intel_shock < 0:
        elasticity += min(0.15, abs(local_intel_shock))

    return max(0.65, min(2.00, elasticity))


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _lead_time_days(target_date, market_context):
    if target_date is None:
        return 14
    try:
        stay_date = pd.Timestamp(target_date).normalize()
    except Exception:
        return 14
    as_of = market_context.get("as_of_timestamp")
    if as_of:
        try:
            return max(0, int((stay_date - pd.Timestamp(as_of).normalize()).days))
        except Exception:
            pass
    return 14


def _historical_pricing_window(target_date, market_context):
    """Return a recent no-leakage ADR slice used to shape pricing policy."""
    try:
        historical = pd.read_csv(DATA_PATH, usecols=["Date", "ADR"], parse_dates=["Date"])
    except (FileNotFoundError, ValueError):
        return None, None

    historical = historical[historical["ADR"].notna() & historical["ADR"].gt(0)].copy()
    if historical.empty:
        return None, None

    latest_actual_date = historical["Date"].max().normalize()
    as_of = market_context.get("as_of_timestamp")
    if as_of:
        try:
            as_of_date = pd.Timestamp(as_of).normalize()
        except Exception:
            as_of_date = latest_actual_date
    elif target_date is not None:
        try:
            as_of_date = min(pd.Timestamp(target_date).normalize() - pd.Timedelta(days=1), latest_actual_date)
        except Exception:
            as_of_date = latest_actual_date
    else:
        as_of_date = latest_actual_date

    as_of_date = min(as_of_date, latest_actual_date)
    start_date = as_of_date - pd.Timedelta(days=BASE_RATE_LOOKBACK_DAYS - 1)
    recent = historical[historical["Date"].between(start_date, as_of_date)].copy()
    if recent.empty:
        recent = historical.copy()
    return recent, as_of_date


def _dynamic_price_policy(target_date, market_context, allowed_premium_pct):
    """Build an explainable recent-history anchor plus adaptive guardrails."""
    recent, as_of_date = _historical_pricing_window(target_date, market_context)
    comp_low = _finite_float(market_context.get("comp_low"))
    comp_median = _finite_float(market_context.get("comp_median"))
    comp_high = _finite_float(market_context.get("comp_high"))
    has_observed_comp_set = (
        str(market_context.get("source_quality", "")).lower() != "legacy_single_rate"
        and int(_finite_float(market_context.get("sample_size")) or 0) >= 2
    )

    if recent is None or recent.empty:
        return {
            "base_price": float(BASE_PRICE),
            "min_price": float(MIN_PRICE),
            "max_price": float(MAX_PRICE),
            "as_of_date": None,
            "lookback_days": 0,
            "recent_median_adr": None,
            "recent_p10_adr": None,
            "recent_p95_adr": None,
            "same_weekday_median_adr": None,
            "has_observed_comp_set": has_observed_comp_set,
            "used_fallback": True,
        }

    recent = recent.copy()
    recent["dow"] = recent["Date"].dt.dayofweek
    recent_median = float(recent["ADR"].median())
    recent_p10 = float(recent["ADR"].quantile(0.10))
    recent_p95 = float(recent["ADR"].quantile(0.95))
    same_weekday_median = recent_median
    if target_date is not None:
        try:
            target_dow = pd.Timestamp(target_date).dayofweek
            weekday_slice = recent.loc[recent["dow"].eq(target_dow), "ADR"]
            if not weekday_slice.empty:
                same_weekday_median = float(weekday_slice.median())
        except Exception:
            pass

    dynamic_base = (
        (DYNAMIC_BASE_DOW_WEIGHT * same_weekday_median)
        + ((1 - DYNAMIC_BASE_DOW_WEIGHT) * recent_median)
    )

    floor_inputs = [float(MIN_PRICE), recent_p10]
    if has_observed_comp_set and comp_low is not None:
        floor_inputs.append(comp_low * DYNAMIC_FLOOR_COMP_LOW_FACTOR)
    dynamic_min = max(floor_inputs)

    ceiling_inputs = [recent_p95, dynamic_base * DYNAMIC_CEILING_BASE_MULTIPLIER]
    if has_observed_comp_set and comp_high is not None:
        ceiling_inputs.append(comp_high * (1 + allowed_premium_pct))
    elif comp_median is not None:
        ceiling_inputs.append(comp_median)
    dynamic_max = min(float(MAX_PRICE), max(ceiling_inputs))
    dynamic_max = max(dynamic_max, dynamic_min)

    return {
        "base_price": _round_money(dynamic_base),
        "min_price": _round_money(dynamic_min),
        "max_price": _round_money(dynamic_max),
        "as_of_date": as_of_date.strftime("%Y-%m-%d") if as_of_date is not None else None,
        "lookback_days": int(len(recent)),
        "recent_median_adr": _round_money(recent_median),
        "recent_p10_adr": _round_money(recent_p10),
        "recent_p95_adr": _round_money(recent_p95),
        "same_weekday_median_adr": _round_money(same_weekday_median),
        "has_observed_comp_set": has_observed_comp_set,
        "used_fallback": False,
    }


def _compression_profile(
    demand_anchor,
    gross_pace_index,
    pickup_trend_index,
    raw_otb_occupancy,
    adjusted_otb_occupancy,
    local_intel_shock,
    lead_time_days,
    market_regime,
):
    raw_occ = demand_anchor if raw_otb_occupancy is None else raw_otb_occupancy
    adjusted_occ = demand_anchor if adjusted_otb_occupancy is None else adjusted_otb_occupancy
    occupancy_score = _clamp01((demand_anchor - 0.55) / 0.40)
    scarcity_score = _clamp01((raw_occ - 0.70) / 0.30)
    gross_pace_score = _clamp01((gross_pace_index - 0.80) / 0.60)
    pickup_score = _clamp01((pickup_trend_index - 0.80) / 0.60)
    lead_score = (
        1.0
        if lead_time_days <= 3
        else 0.75
        if lead_time_days <= 7
        else 0.50
        if lead_time_days <= 14
        else 0.25
        if lead_time_days <= 30
        else 0.0
    )
    event_score = _clamp01(max(local_intel_shock, 0.0) / 0.15)
    retention_ratio = adjusted_occ / raw_occ if raw_occ and raw_occ > 0 else 1.0
    retention_confidence = _clamp01((retention_ratio - 0.55) / 0.35)
    market_bonus = 0.08 if market_regime == "market_wide_sellout" else 0.04 if market_regime == "event_compression" else 0.0

    score = (
        (0.40 * occupancy_score)
        + (0.20 * scarcity_score)
        + (0.12 * gross_pace_score)
        + (0.08 * pickup_score)
        + (0.10 * lead_score)
        + (0.10 * event_score)
    )
    score = _clamp01((score * (0.75 + (0.25 * retention_confidence))) + market_bonus)

    if score >= 0.80:
        allowed_premium_pct = 0.18
        regime = "last_room"
    elif score >= 0.60:
        allowed_premium_pct = 0.12
        regime = "strong_compression"
    elif score >= 0.35:
        allowed_premium_pct = 0.06
        regime = "healthy_demand"
    else:
        allowed_premium_pct = 0.02
        regime = "soft_market"

    if market_regime == "market_wide_sellout":
        allowed_premium_pct = min(0.20, allowed_premium_pct + 0.02)
    elif market_regime == "event_compression":
        allowed_premium_pct = min(0.20, allowed_premium_pct + 0.01)

    return {
        "compression_score": round(score, 4),
        "allowed_premium_pct": round(allowed_premium_pct, 4),
        "market_position_regime": regime,
        "retention_ratio": round(retention_ratio, 4),
        "lead_time_days": int(lead_time_days),
    }


def _estimate_expected_occupancy(candidate_price, demand_anchor, reference_price, elasticity, market_context, allowed_premium_pct):
    price_ratio = (candidate_price - reference_price) / max(reference_price, 1.0)
    expected_occ = demand_anchor * np.exp(-elasticity * price_ratio)

    comp_median = market_context.get("comp_median")
    comp_high = market_context.get("comp_high")
    if comp_median is not None and comp_median > 0:
        headroom_price = comp_median * (1 + allowed_premium_pct)
        if candidate_price <= comp_median:
            below_median_gap = (comp_median - candidate_price) / comp_median
            expected_occ *= min(1.08, 1 + (0.08 * below_median_gap))
        elif candidate_price <= headroom_price:
            denominator = max(headroom_price - comp_median, 1.0)
            within_headroom = (candidate_price - comp_median) / denominator
            expected_occ *= max(0.90, 1 - (0.10 * within_headroom))
        elif comp_high is not None and candidate_price <= comp_high:
            denominator = max(comp_high - headroom_price, 1.0)
            beyond_headroom = (candidate_price - headroom_price) / denominator
            expected_occ *= max(0.75, 0.90 - (0.15 * beyond_headroom))
        else:
            high_reference = max(comp_high or headroom_price, 1.0)
            above_high_gap = max(0.0, (candidate_price - high_reference) / high_reference)
            expected_occ *= max(0.45, 0.75 - (0.70 * above_high_gap))

    return max(0.0, min(1.0, float(expected_occ)))


def _manual_review_flags(
    final_price,
    reference_price,
    market_context,
    occupancy,
    pickup_trend_index,
    manual_event_text,
    local_intel_shock,
    *,
    sold_out=False,
    raw_otb_occupancy=None,
    adjusted_otb_occupancy=None,
):
    flags = []
    competitor_price = market_context.get("comp_median")
    comp_high = market_context.get("comp_high")
    if competitor_price is None:
        flags.append("Competitor price unavailable; optimizer used internal reference pricing only.")
    else:
        competitor_gap = ((final_price - competitor_price) / competitor_price) * 100
        if competitor_gap > 25:
            flags.append(f"Recommended ADR is {competitor_gap:.1f}% above competitor median; review market positioning before publishing.")
        elif competitor_gap < -20 and occupancy >= 0.80:
            flags.append(f"Recommended ADR is {abs(competitor_gap):.1f}% below competitor median despite strong demand; check for underpricing.")
        if comp_high and final_price > comp_high:
            high_gap = ((final_price - comp_high) / comp_high) * 100
            flags.append(f"Recommended ADR is {high_gap:.1f}% above the high end of the comp set; confirm the premium before publishing.")

    reference_gap = ((final_price - reference_price) / max(reference_price, 1.0)) * 100
    if abs(reference_gap) > 20:
        flags.append(f"Recommended ADR moved {reference_gap:+.1f}% from the reference price.")

    if occupancy >= 0.92 and pickup_trend_index >= 1.2:
        flags.append("High compression and fast pickup detected; monitor guest perception and remaining inventory.")

    if sold_out and _has_material_retention_gap(raw_otb_occupancy, adjusted_otb_occupancy):
        flags.append(
            "Raw OTB is sold out while retained occupancy is materially lower after cancellation risk; "
            "verify cancellation assumptions and remaining-room strategy before publishing."
        )

    if manual_event_text and local_intel_shock:
        flags.append("Local intel was applied to priced demand; confirm the event impact is still appropriate before publishing.")
    elif manual_event_text:
        flags.append("Local intel was supplied as context only and was not applied to priced demand.")

    return flags


def calculate_recommended_price(
    occupancy,
    day_name,
    target_date=None,
    competitor_price=None,
    market_context=None,
    return_breakdown=False,
    pre_shock_occupancy=None,
    manual_shock=0.0,
    local_intel_shock=0.0,
    booking_velocity=1.0,
    gross_pace_index=None,
    retained_pace_index=None,
    pickup_trend_index=None,
    pricing_pace_index=None,
    manual_event_text="",
    raw_otb_occupancy=None,
    adjusted_otb_occupancy=None,
    expected_cancellations=0.0,
):
    """Select the allowed ADR with the highest expected room revenue."""
    occupancy = max(0.0, min(1.0, float(occupancy)))
    organic_occupancy = occupancy if pre_shock_occupancy is None else max(0.0, min(1.0, float(pre_shock_occupancy)))
    pace_signals = _resolve_pace_signals(
        booking_velocity=booking_velocity,
        gross_pace_index=gross_pace_index,
        retained_pace_index=retained_pace_index,
        pickup_trend_index=pickup_trend_index,
        pricing_pace_index=pricing_pace_index,
    )
    booking_velocity = pace_signals["booking_velocity"]
    gross_pace_index = pace_signals["gross_pace_index"]
    retained_pace_index = pace_signals["retained_pace_index"]
    pickup_trend_index = pace_signals["pickup_trend_index"]
    pricing_pace_index = pace_signals["pricing_pace_index"]
    market_context = normalize_market_context(market_context, competitor_price)
    competitor_price = market_context.get("comp_median")
    raw_otb_occupancy = _finite_float(raw_otb_occupancy)
    adjusted_otb_occupancy = _finite_float(adjusted_otb_occupancy)
    if raw_otb_occupancy is not None:
        raw_otb_occupancy = max(0.0, min(1.0, raw_otb_occupancy))
    if adjusted_otb_occupancy is not None:
        adjusted_otb_occupancy = max(0.0, min(1.0, adjusted_otb_occupancy))
    expected_cancellations = max(0.0, float(expected_cancellations or 0.0))
    sold_out = _is_sold_out(raw_otb_occupancy)
    applied_rules = []
    components = []

    manual_shock = float(manual_shock or 0.0)
    local_intel_shock = float(local_intel_shock or 0.0)
    if pre_shock_occupancy is not None and manual_shock == 0 and local_intel_shock == 0 and occupancy != organic_occupancy:
        manual_shock = occupancy - organic_occupancy

    manual_occ = max(0.0, min(1.0, organic_occupancy + manual_shock))
    demand_anchor = max(0.0, min(1.0, manual_occ + local_intel_shock))
    if pricing_pace_index > 1.0:
        demand_anchor *= 1 + min(0.12, (pricing_pace_index - 1.0) * 0.20)
    elif pricing_pace_index < 1.0:
        demand_anchor *= 1 - min(0.12, (1.0 - pricing_pace_index) * 0.25)
    demand_anchor = max(0.0, min(1.0, demand_anchor))

    elasticity = _dynamic_elasticity(demand_anchor, pricing_pace_index, local_intel_shock)
    lead_time_days = _lead_time_days(target_date, market_context)
    compression = _compression_profile(
        demand_anchor,
        gross_pace_index,
        pickup_trend_index,
        raw_otb_occupancy,
        adjusted_otb_occupancy,
        local_intel_shock,
        lead_time_days,
        market_context.get("market_regime"),
    )
    pricing_policy = _dynamic_price_policy(
        target_date,
        market_context,
        compression["allowed_premium_pct"],
    )
    base_price = pricing_policy["base_price"]
    min_price = pricing_policy["min_price"]
    max_price = pricing_policy["max_price"]
    recommended_price = float(base_price)

    _add_component(
        components,
        "Base rate",
        0,
        recommended_price,
        (
            f"Recent-history base rate from the last {pricing_policy['lookback_days']} days: "
            f"{DYNAMIC_BASE_DOW_WEIGHT * 100:.0f}% same-weekday median ADR "
            f"(${pricing_policy['same_weekday_median_adr']:.2f}) and "
            f"{(1 - DYNAMIC_BASE_DOW_WEIGHT) * 100:.0f}% overall median ADR "
            f"(${pricing_policy['recent_median_adr']:.2f})."
            if not pricing_policy["used_fallback"]
            else "Fallback public base rate used because recent ADR history was unavailable."
        ),
    )
    reference_price = float(base_price)
    if competitor_price is not None:
        market_pull = 0.20 + (0.60 * compression["compression_score"])
        compression_lift = competitor_price * compression["allowed_premium_pct"] * 0.50 * compression["compression_score"]
        reference_price = (
            (base_price * (1 - market_pull))
            + (competitor_price * market_pull)
            + compression_lift
        )
        reference_price = max(min_price, min(max_price, reference_price))

    candidates = []
    for candidate in _candidate_prices(min_price, max_price):
        expected_occupancy = _estimate_expected_occupancy(
            candidate,
            demand_anchor,
            reference_price,
            elasticity,
            market_context,
            compression["allowed_premium_pct"],
        )
        expected_rooms = expected_occupancy * BASE_CAPACITY
        expected_revenue = candidate * expected_rooms
        candidates.append(
            {
                "price": _round_money(candidate),
                "expected_occupancy": round(expected_occupancy, 4),
                "expected_rooms": round(expected_rooms, 2),
                "expected_revenue": _round_money(expected_revenue),
            }
        )

    selected = max(candidates, key=lambda row: (row["expected_revenue"], row["price"]))
    optimized_price = selected["price"]
    recommended_price = optimized_price
    applied_rules.append("Candidate Price Optimizer")

    _add_component(
        components,
        "Demand anchor",
        recommended_price,
        recommended_price,
        f"Priced demand anchor is {round(demand_anchor * 100)}% after forecast/current occupancy, manual demand, local intel, and composite pace signals.",
    )
    _add_component(
        components,
        "Reference price",
        base_price,
        reference_price,
        "Reference price blends the internal base rate with comp-set context and demand compression when available.",
    )
    _add_component(
        components,
        "Candidate optimization",
        reference_price,
        optimized_price,
        f"Selected the candidate ADR with the highest expected room revenue using elasticity {elasticity:.2f}.",
    )
    _add_component(
        components,
        "Competitor signal",
        optimized_price,
        optimized_price,
        (
            f"Comp-set median ${competitor_price:.2f} and high ${market_context['comp_high']:.2f} were used as soft demand signals; "
            f"{compression['market_position_regime'].replace('_', ' ')} allows up to {compression['allowed_premium_pct'] * 100:.0f}% premium over median."
            if competitor_price is not None
            else "No finite competitor price was available."
        ),
    )

    sold_out_floor_price = None
    sold_out_floor_applied = False
    if sold_out:
        floor_before = recommended_price
        sold_out_floor_price = competitor_price
        if competitor_price is not None:
            recommended_price = max(recommended_price, competitor_price)
            sold_out_floor_applied = recommended_price > floor_before
            if sold_out_floor_applied:
                applied_rules.append("Sold-Out Competitor Median Floor")
            floor_explanation = (
                "Raw OTB reached capacity, so scarce-inventory pricing cannot fall below competitor median."
                if sold_out_floor_applied
                else "Raw OTB reached capacity; optimizer ADR already meets or exceeds competitor median."
            )
        else:
            floor_explanation = "Raw OTB reached capacity, but no competitor price was available for a parity floor."
        _add_component(components, "Sold-out floor", floor_before, recommended_price, floor_explanation)

    before_guardrail = recommended_price
    recommended_price = max(min_price, min(max_price, recommended_price))
    _add_component(
        components,
        "Final recommendation",
        before_guardrail,
        recommended_price,
        f"Dynamic safety bounds enforced between ${min_price:.2f} and ${max_price:.2f}.",
    )

    final_price = round(recommended_price, 2)
    final_expected_occupancy = _estimate_expected_occupancy(
        final_price,
        demand_anchor,
        reference_price,
        elasticity,
        market_context,
        compression["allowed_premium_pct"],
    )
    final_expected_rooms = final_expected_occupancy * BASE_CAPACITY
    final_expected_revenue = final_price * final_expected_rooms
    review_flags = _manual_review_flags(
        final_price,
        reference_price,
        market_context,
        demand_anchor,
        pickup_trend_index,
        manual_event_text,
        local_intel_shock,
        sold_out=sold_out,
        raw_otb_occupancy=raw_otb_occupancy,
        adjusted_otb_occupancy=adjusted_otb_occupancy,
    )
    if not return_breakdown:
        return final_price, applied_rules + review_flags

    breakdown = {
        "base_price": _round_money(base_price),
        "static_base_price": _round_money(BASE_PRICE),
        "reference_price": _round_money(reference_price),
        "final_price": final_price,
        "dynamic_min_price": _round_money(min_price),
        "dynamic_max_price": _round_money(max_price),
        "pricing_policy": pricing_policy,
        "occupancy_used": round(occupancy, 4),
        "pre_shock_occupancy": round(organic_occupancy, 4),
        "raw_otb_occupancy": round(raw_otb_occupancy, 4) if raw_otb_occupancy is not None else None,
        "adjusted_otb_occupancy": round(adjusted_otb_occupancy, 4) if adjusted_otb_occupancy is not None else None,
        "expected_cancellations": _round_money(expected_cancellations),
        "sold_out": sold_out,
        "pricing_regime": "sold_out_protect_rate" if sold_out else "normal",
        "sold_out_floor_price": _round_money(sold_out_floor_price) if sold_out_floor_price is not None else None,
        "sold_out_floor_applied": sold_out_floor_applied,
        "material_retention_gap": _has_material_retention_gap(raw_otb_occupancy, adjusted_otb_occupancy),
        "pricing_occupancy": round(organic_occupancy, 4),
        "demand_anchor": round(demand_anchor, 4),
        "manual_shock": round(manual_shock, 4),
        "local_intel_shock": round(local_intel_shock, 4),
        "booking_velocity": round(booking_velocity, 4),
        "gross_pace_index": round(gross_pace_index, 4),
        "retained_pace_index": round(retained_pace_index, 4),
        "pickup_trend_index": round(pickup_trend_index, 4),
        "pricing_pace_index": round(pricing_pace_index, 4),
        "elasticity": round(elasticity, 4),
        "day_name": day_name,
        "competitor_price": _round_money(competitor_price) if competitor_price is not None else None,
        "market_context": market_context,
        "compression_score": compression["compression_score"],
        "allowed_premium_pct": compression["allowed_premium_pct"],
        "market_position_regime": compression["market_position_regime"],
        "retention_ratio": compression["retention_ratio"],
        "lead_time_days": compression["lead_time_days"],
        "optimizer_candidates": candidates,
        "selected_candidate": selected,
        "final_recommendation": {
            "price": final_price,
            "expected_occupancy": round(final_expected_occupancy, 4),
            "expected_rooms": round(final_expected_rooms, 2),
            "expected_revenue": _round_money(final_expected_revenue),
        },
        "expected_rooms": round(final_expected_rooms, 2),
        "expected_revenue": _round_money(final_expected_revenue),
        "competitor_gap_pct": _round_money(((final_price - competitor_price) / competitor_price) * 100) if competitor_price else None,
        "comp_median_gap_pct": _round_money(((final_price - competitor_price) / competitor_price) * 100) if competitor_price else None,
        "comp_high_gap_pct": _round_money(((final_price - market_context["comp_high"]) / market_context["comp_high"]) * 100) if market_context.get("comp_high") else None,
        "review_flags": review_flags,
        "components": components,
        "applied_rules": applied_rules + review_flags,
    }
    return final_price, applied_rules + review_flags, breakdown


def generate_pricing_report():
    """Read the demand forecast and apply pricing logic to all future dates."""
    try:
        df = pd.read_csv(FORECAST_OUTPUT_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day_Name"] = df["Date"].dt.day_name()
        df["Recommended_Price"] = df.apply(
            lambda row: calculate_recommended_price(
                row["Forecasted_Occupancy"],
                row["Day_Name"],
                row["Date"],
                competitor_price=row.get("Competitor_Rate", None),
            )[0],
            axis=1,
        )
        df["Forecasted_Revenue"] = df["Recommended_Price"] * (df["Forecasted_Occupancy"] * BASE_CAPACITY)
        print(f"Pricing Engine successfully processed {len(df)} days.")
        return df
    except FileNotFoundError:
        print("Error: Forecast file not found. Run demand_forecast.py first.")
        return None


if __name__ == "__main__":
    report = generate_pricing_report()
    if report is not None:
        print(report[["Date", "Day_Name", "Forecasted_Occupancy", "Recommended_Price"]].head())
