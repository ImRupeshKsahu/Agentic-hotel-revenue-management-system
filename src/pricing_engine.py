import pandas as pd
import numpy as np
from config import (
    FORECAST_OUTPUT_PATH,
    MIN_PRICE,
    MAX_PRICE,
    BASE_PRICE,
    BASE_CAPACITY,
    PRICE_STEP,
)


def _round_money(value):
    return round(float(value), 2)


def _add_component(components, label, before, after, explanation):
    before = float(before)
    after = float(after)
    components.append({
        "driver": label,
        "price_before": _round_money(before),
        "adjustment": _round_money(after - before),
        "price_after": _round_money(after),
        "explanation": explanation,
    })


def _finite_float(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def _candidate_prices():
    start = int(np.ceil(MIN_PRICE / PRICE_STEP) * PRICE_STEP)
    end = int(np.floor(MAX_PRICE / PRICE_STEP) * PRICE_STEP)
    return [float(price) for price in range(start, end + PRICE_STEP, PRICE_STEP)]


def _dynamic_elasticity(occupancy, booking_velocity, local_intel_shock):
    """Conservative elasticity proxy for a PoC optimizer.

    Higher compression makes demand less price sensitive; weak pace makes it
    more sensitive. This is intentionally transparent until a learned demand
    model replaces it.
    """
    elasticity = 1.35
    if occupancy >= 0.90:
        elasticity -= 0.35
    elif occupancy >= 0.80:
        elasticity -= 0.20
    elif occupancy < 0.55:
        elasticity += 0.30

    if booking_velocity >= 1.25:
        elasticity -= 0.20
    elif booking_velocity <= 0.80:
        elasticity += 0.25

    if local_intel_shock > 0:
        elasticity -= min(0.15, local_intel_shock)
    elif local_intel_shock < 0:
        elasticity += min(0.15, abs(local_intel_shock))

    return max(0.65, min(2.00, elasticity))


def _estimate_expected_occupancy(candidate_price, demand_anchor, reference_price, elasticity, competitor_price):
    price_ratio = (candidate_price - reference_price) / max(reference_price, 1.0)
    expected_occ = demand_anchor * np.exp(-elasticity * price_ratio)

    if competitor_price is not None and competitor_price > 0:
        competitor_gap = (candidate_price - competitor_price) / competitor_price
        if competitor_gap > 0:
            expected_occ *= max(0.50, 1 - 0.65 * competitor_gap)
        else:
            expected_occ *= min(1.10, 1 + 0.12 * abs(competitor_gap))

    return max(0.0, min(1.0, float(expected_occ)))


def _manual_review_flags(final_price, reference_price, competitor_price, occupancy, booking_velocity, manual_event_text, local_intel_shock):
    flags = []
    if competitor_price is None:
        flags.append("Competitor price unavailable; optimizer used internal reference pricing only.")
    else:
        competitor_gap = ((final_price - competitor_price) / competitor_price) * 100
        if competitor_gap > 25:
            flags.append(f"Recommended ADR is {competitor_gap:.1f}% above competitor; review market positioning before publishing.")
        elif competitor_gap < -20 and occupancy >= 0.80:
            flags.append(f"Recommended ADR is {abs(competitor_gap):.1f}% below competitor despite strong demand; check for underpricing.")

    reference_gap = ((final_price - reference_price) / max(reference_price, 1.0)) * 100
    if abs(reference_gap) > 20:
        flags.append(f"Recommended ADR moved {reference_gap:+.1f}% from the reference price.")

    if occupancy >= 0.92 and booking_velocity >= 1.2:
        flags.append("High compression and fast pickup detected; monitor guest perception and remaining inventory.")

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
    return_breakdown=False,
    pre_shock_occupancy=None,
    manual_shock=0.0,
    local_intel_shock=0.0,
    booking_velocity=1.0,
    manual_event_text="",
    raw_otb_occupancy=None,
    adjusted_otb_occupancy=None,
    expected_cancellations=0.0,
):
    """
    Core Pricing Logic Engine.

    The candidate-price optimizer estimates expected demand for each allowed price and selects the ADR with
    the highest expected room revenue. When two candidates tie on revenue, the current policy favors the
    higher price; this is intentional, but may be flipped during distressed periods if the business prefers
    occupancy protection over ADR.
    """
    base_price = BASE_PRICE
    occupancy = max(0.0, min(1.0, float(occupancy)))
    organic_occupancy = occupancy if pre_shock_occupancy is None else max(0.0, min(1.0, float(pre_shock_occupancy)))
    booking_velocity = max(0.1, float(booking_velocity or 1.0))
    competitor_price = _finite_float(competitor_price)
    raw_otb_occupancy = _finite_float(raw_otb_occupancy)
    adjusted_otb_occupancy = _finite_float(adjusted_otb_occupancy)
    if raw_otb_occupancy is not None:
        raw_otb_occupancy = max(0.0, min(1.0, raw_otb_occupancy))
    if adjusted_otb_occupancy is not None:
        adjusted_otb_occupancy = max(0.0, min(1.0, adjusted_otb_occupancy))
    expected_cancellations = max(0.0, float(expected_cancellations or 0.0))
    recommended_price = float(base_price)
    applied_rules = []
    components = []

    _add_component(
        components,
        "Base rate",
        0,
        recommended_price,
        "Starting public rate before demand, calendar, market, or safety adjustments.",
    )

    manual_shock = float(manual_shock or 0.0)
    local_intel_shock = float(local_intel_shock or 0.0)
    if pre_shock_occupancy is not None and manual_shock == 0 and local_intel_shock == 0 and occupancy != organic_occupancy:
        manual_shock = occupancy - organic_occupancy

    manual_occ = max(0.0, min(1.0, organic_occupancy + manual_shock))
    demand_anchor = max(0.0, min(1.0, manual_occ + local_intel_shock))
    if booking_velocity > 1.0:
        demand_anchor *= 1 + min(0.12, (booking_velocity - 1.0) * 0.20)
    elif booking_velocity < 1.0:
        demand_anchor *= 1 - min(0.12, (1.0 - booking_velocity) * 0.25)
    demand_anchor = max(0.0, min(1.0, demand_anchor))

    reference_price = float(base_price)
    if competitor_price is not None:
        reference_price = max(MIN_PRICE, min(MAX_PRICE, (base_price * 0.60) + (competitor_price * 0.40)))

    elasticity = _dynamic_elasticity(demand_anchor, booking_velocity, local_intel_shock)
    candidates = []
    for candidate in _candidate_prices():
        expected_occupancy = _estimate_expected_occupancy(
            candidate,
            demand_anchor,
            reference_price,
            elasticity,
            competitor_price,
        )
        expected_rooms = expected_occupancy * BASE_CAPACITY
        expected_revenue = candidate * expected_rooms
        candidates.append({
            "price": _round_money(candidate),
            "expected_occupancy": round(expected_occupancy, 4),
            "expected_rooms": round(expected_rooms, 2),
            "expected_revenue": _round_money(expected_revenue),
        })

    # Revenue is the primary objective. On an exact tie, favor the higher ADR for now.
    # This policy is explicit because the business may want to reverse it in distressed periods.
    selected = max(candidates, key=lambda row: (row["expected_revenue"], row["price"]))
    optimized_price = selected["price"]
    recommended_price = optimized_price
    applied_rules.append("Candidate Price Optimizer")

    _add_component(
        components,
        "Demand anchor",
        recommended_price,
        recommended_price,
        f"Priced demand anchor is {round(demand_anchor * 100)}% after forecast/current occupancy, manual demand, local intel, and booking pace.",
    )
    _add_component(
        components,
        "Reference price",
        base_price,
        reference_price,
        "Reference price blends the internal base rate with competitor context when available.",
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
        "Competitor price was used as a soft demand signal, not a hard ceiling." if competitor_price is not None else "No finite competitor price was available.",
    )

    # Safety Guardrails (From Config)
    before_guardrail = recommended_price
    recommended_price = max(MIN_PRICE, min(MAX_PRICE, recommended_price))
    _add_component(
        components,
        "Final recommendation",
        before_guardrail,
        recommended_price,
        f"Safety bounds enforced between ${MIN_PRICE:.2f} and ${MAX_PRICE:.2f}.",
    )

    final_price = round(recommended_price, 2)
    review_flags = _manual_review_flags(
        final_price,
        reference_price,
        competitor_price,
        demand_anchor,
        booking_velocity,
        manual_event_text,
        local_intel_shock,
    )
    if not return_breakdown:
        return final_price, applied_rules + review_flags

    breakdown = {
        "base_price": _round_money(base_price),
        "reference_price": _round_money(reference_price),
        "final_price": final_price,
        "occupancy_used": round(occupancy, 4),
        "pre_shock_occupancy": round(organic_occupancy, 4),
        "raw_otb_occupancy": round(raw_otb_occupancy, 4) if raw_otb_occupancy is not None else None,
        "adjusted_otb_occupancy": round(adjusted_otb_occupancy, 4) if adjusted_otb_occupancy is not None else None,
        "expected_cancellations": _round_money(expected_cancellations),
        "pricing_occupancy": round(organic_occupancy, 4),
        "demand_anchor": round(demand_anchor, 4),
        "manual_shock": round(manual_shock, 4),
        "local_intel_shock": round(local_intel_shock, 4),
        "booking_velocity": round(booking_velocity, 4),
        "elasticity": round(elasticity, 4),
        "day_name": day_name,
        "competitor_price": _round_money(competitor_price) if competitor_price is not None else None,
        "optimizer_candidates": candidates,
        "selected_candidate": selected,
        "expected_rooms": selected["expected_rooms"],
        "expected_revenue": selected["expected_revenue"],
        "competitor_gap_pct": _round_money(((final_price - competitor_price) / competitor_price) * 100) if competitor_price else None,
        "review_flags": review_flags,
        "components": components,
        "applied_rules": applied_rules + review_flags,
    }
    return final_price, applied_rules + review_flags, breakdown

def generate_pricing_report():
    """
    Reads the demand forecast and applies pricing logic to all 30 days.
    """
    try:
        # Load the latest forecast
        df = pd.read_csv(FORECAST_OUTPUT_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add helper columns for logic
        df['Day_Name'] = df['Date'].dt.day_name()

        # Apply the pricing function row by row
        df['Recommended_Price'] = df.apply(
            lambda row: calculate_recommended_price(
                row['Forecasted_Occupancy'], 
                row['Day_Name'],
                row['Date'],
                competitor_price=row.get('Competitor_Rate', None)
            )[0], axis=1
        )

        # Calculate Revenue Forecast (Price * Predicted Rooms)
        df['Forecasted_Revenue'] = df['Recommended_Price'] * (df['Forecasted_Occupancy'] * BASE_CAPACITY )

        print(f"Pricing Engine successfully processed {len(df)} days.")
        return df

    except FileNotFoundError:
        print("Error: Forecast file not found. Run demand_forecast.py first.")
        return None

if __name__ == "__main__":
    report = generate_pricing_report()
    if report is not None:
        print(report[['Date', 'Day_Name', 'Forecasted_Occupancy', 'Recommended_Price']].head())
