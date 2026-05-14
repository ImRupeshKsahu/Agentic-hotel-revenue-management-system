import pandas as pd
import numpy as np
from config import FORECAST_OUTPUT_PATH, MIN_PRICE, MAX_PRICE, BASE_PRICE, BASE_CAPACITY

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


def calculate_recommended_price(
    occupancy,
    day_name,
    target_date=None,
    competitor_price=None,
    return_breakdown=False,
    pre_shock_occupancy=None,
    manual_shock=0.0,
    local_intel_shock=0.0,
):
    """
    Core Pricing Logic Engine
    Rules:
    1. Saturday Yield: +15% if Occupancy > 70%
    2. Sunday Slump: -10% if Occupancy < 65% (To capture check-out loss)
    3. Business Mid-week: +10% on Tue/Wed if Occupancy > 80%
    4. Competitive Matching: Never be more than 20% higher than competitor
    """
    base_price = BASE_PRICE
    occupancy = max(0.0, min(1.0, float(occupancy)))
    organic_occupancy = occupancy if pre_shock_occupancy is None else max(0.0, min(1.0, float(pre_shock_occupancy)))
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

    # Rule 1: Occupancy-based Dynamic Pricing (The "Multiplier")
    # Using a simple linear elasticity: every 10% occupancy above 50% adds $10
    organic_lift = max(0.0, organic_occupancy - 0.50) * 100
    manual_shock = float(manual_shock or 0.0)
    local_intel_shock = float(local_intel_shock or 0.0)
    if pre_shock_occupancy is not None and manual_shock == 0 and local_intel_shock == 0 and occupancy != organic_occupancy:
        manual_shock = occupancy - organic_occupancy
    if organic_lift != 0:
        before = recommended_price
        recommended_price += organic_lift
        _add_component(
            components,
            "Occupancy lift",
            before,
            recommended_price,
            f"Forecast/current occupancy pressure at {round(organic_occupancy * 100)}%.",
        )
        applied_rules.append(f"Demand-Based Scaling (Occ: {round(occupancy*100)}%)")
    else:
        _add_component(
            components,
            "Occupancy lift",
            recommended_price,
            recommended_price,
            "Occupancy is at or below the 50% demand threshold.",
        )

    if pre_shock_occupancy is not None and (manual_shock != 0 or local_intel_shock != 0):
        manual_occ = max(0.0, min(1.0, organic_occupancy + manual_shock))
        manual_lift = (max(0.0, manual_occ - 0.50) - max(0.0, organic_occupancy - 0.50)) * 100
        before = recommended_price
        recommended_price += manual_lift
        _add_component(
            components,
            "Manual demand adjustment",
            before,
            recommended_price,
            f"Manual slider moved priced occupancy from {round(organic_occupancy * 100)}% to {round(manual_occ * 100)}%.",
        )

        local_occ = max(0.0, min(1.0, manual_occ + local_intel_shock))
        local_lift = (max(0.0, local_occ - 0.50) - max(0.0, manual_occ - 0.50)) * 100
        before = recommended_price
        recommended_price += local_lift
        _add_component(
            components,
            "Local intel adjustment",
            before,
            recommended_price,
            f"Applied local intel moved priced occupancy from {round(manual_occ * 100)}% to {round(local_occ * 100)}%.",
        )
    else:
        _add_component(
            components,
            "Manual demand adjustment",
            recommended_price,
            recommended_price,
            "No manual slider adjustment was applied.",
        )
        _add_component(
            components,
            "Local intel adjustment",
            recommended_price,
            recommended_price,
            "Local intel was context-only and was not included in the baseline price.",
        )

    # Rule 2: Saturday Yield
    if day_name == "Saturday" and occupancy > 0.70:
        before = recommended_price
        recommended_price *= 1.15
        _add_component(
            components,
            "Day-of-week effect",
            before,
            recommended_price,
            "Saturday yield premium applied because demand is above 70%.",
        )
        applied_rules.append("Saturday Yield (+15%)")

    # Rule 3: Sunday Slump Correction
    elif day_name == "Sunday" and occupancy < 0.65:
        before = recommended_price
        recommended_price *= 0.90
        _add_component(
            components,
            "Day-of-week effect",
            before,
            recommended_price,
            "Sunday slump discount applied because demand is below 65%.",
        )
        applied_rules.append("Sunday Slump Discount (-10%) (historically it has been observed that Sunday occupancy drops down and people checking out on Sunday)")

    # Rule 4: Business Peak (Tue/Wed logic)
    elif day_name in ["Tuesday", "Wednesday"] and occupancy > 0.80:
        before = recommended_price
        recommended_price *= 1.10
        _add_component(
            components,
            "Day-of-week effect",
            before,
            recommended_price,
            "Tuesday/Wednesday business premium applied because demand is above 80%.",
        )
        applied_rules.append(f"Business Peak on Tuesday/Wednesday applied")
    else:
        _add_component(
            components,
            "Day-of-week effect",
            recommended_price,
            recommended_price,
            "No day-of-week premium or discount was triggered.",
        )

    # Rule 5: Competitor Benchmarking
    if competitor_price is not None and np.isfinite(float(competitor_price)):
        # If we are 20% higher than the market, we risk losing volume
        ceiling = float(competitor_price) * 1.20
        before = recommended_price
        if recommended_price > ceiling:
            recommended_price = ceiling
        _add_component(
            components,
            "Competitor cap",
            before,
            recommended_price,
            f"Competitor ceiling checked at ${round(ceiling, 2)}.",
        )
        applied_rules.append(f"Competitor Price Ceiling Applied (Capped at ${round(ceiling, 2)})")
    else:
        _add_component(
            components,
            "Competitor cap",
            recommended_price,
            recommended_price,
            "No finite competitor price was available.",
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
    if not return_breakdown:
        return final_price, applied_rules

    breakdown = {
        "base_price": _round_money(base_price),
        "final_price": final_price,
        "occupancy_used": round(occupancy, 4),
        "pre_shock_occupancy": round(organic_occupancy, 4),
        "manual_shock": round(manual_shock, 4),
        "local_intel_shock": round(local_intel_shock, 4),
        "day_name": day_name,
        "competitor_price": _round_money(competitor_price) if competitor_price is not None and np.isfinite(float(competitor_price)) else None,
        "components": components,
        "applied_rules": applied_rules,
    }
    return final_price, applied_rules, breakdown

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
                row['Date']
            ), axis=1
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
