import pandas as pd
import pickle
import os
from project_core.config import MODEL_PATH, BASE_CAPACITY, DATA_PATH
from pricing_core.engine import calculate_recommended_price

class ScenarioEngine:
    def __init__(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        except Exception:
            self.model = None
        self.market_data = pd.read_csv(DATA_PATH)
        self.market_data['Date'] = pd.to_datetime(self.market_data['Date'], dayfirst=True, errors='coerce')

    def run_scenario(self, target_date, event_status=False, demand_shock=0.0, comp_rate_change=0.0):
        """
        Runs a deep simulation including external shocks and competitor moves.
        :param demand_shock: Float (e.g., -0.2 for 20% drop, 0.1 for 10% increase)
        :param comp_rate_change: Float (e.g., 0.15 for 15% increase)
        """
        target_dt = pd.to_datetime(target_date)

        # Find Baseline Competitor Rate for this date
        # If date not in history (future), we use the average historical competitor rate
        try:
            base_comp_price = self.market_data[self.market_data['Date'] == target_dt]['Competitor_Rate'].values[0]
        except IndexError:
            base_comp_price = self.market_data['Competitor_Rate'].mean()


        # --- STEP 1: CALCULATE PURE BASELINE (No Event, No Shock) ---
        if self.model is not None:
            base_row = pd.DataFrame({'ds': [target_dt], 'Local_Event': [0]})
            base_forecast = self.model.predict(base_row)
            baseline_occ = base_forecast['yhat'].values[0]
        else:
            baseline_occ = self.market_data['Occupancy_Rate'].dropna().tail(28).mean()

        # --- STEP 2: CALCULATE EVENT IMPACT ---
        # We predict again WITH the event to see the 'AI-driven' jump
        if self.model is not None:
            event_row = pd.DataFrame({'ds': [target_dt], 'Local_Event': [1 if event_status else 0]})
            event_forecast = self.model.predict(event_row)
            ai_event_occ = event_forecast['yhat'].values[0]
        else:
            ai_event_occ = baseline_occ + (0.1 if event_status else 0)

        event_delta = ai_event_occ - baseline_occ if event_status else 0

        # --- STEP 3: APPLY MANUAL DEMAND SHOCK ---
        final_simulated_occ = max(0, min(1, ai_event_occ + demand_shock))

        # --- STEP 4: COMPETITOR & PRICING ---
        # (Lookup market_rate from your unified file as discussed before)
        # ... [market_rate lookup code] ...
        new_comp_price = base_comp_price * (1 + comp_rate_change)

        day_name = target_dt.day_name()
        
        simulated_price, logic_flags  = calculate_recommended_price(
            occupancy= final_simulated_occ,
            day_name=day_name,
            competitor_price=new_comp_price
        )

        # 4. Calculate Revenue Metrics
        rooms_sold = round(final_simulated_occ* BASE_CAPACITY)
        total_revenue = round(rooms_sold * simulated_price, 2)
        
        return {
            "date": target_date,
            "base_occupancy": round(baseline_occ, 2),
            "event_impact": round(event_delta, 2),
            "manual_shock": demand_shock,
            "final_occ": round(final_simulated_occ, 2),
            "comp_price_change_pct": comp_rate_change * 100,
            "final_suggested_price": simulated_price,
            "original_comp_price": round(base_comp_price, 2),
            "new_comp_price": round(new_comp_price, 2),
            "projected_revenue": total_revenue,
            "logic_flags": logic_flags,
            "rooms_to_sell": rooms_sold
        }
# --- Example Usage for Testing ---
if __name__ == "__main__":
    engine = ScenarioEngine()
    
    # Simulate a 'Normal' Sunday
    normal_sun = engine.run_scenario("2017-09-10", event_status=False)
    print(f"Normal Sunday: {normal_sun}")

    # Simulate an 'Event' Sunday (e.g., a massive festival)
    event_sun = engine.run_scenario("2017-09-10", event_status =True, demand_shock=-0.3, comp_rate_change=-0.4)
    print(f"Event Sunday: {event_sun}")
    
    # Calculate Delta for GenAI Explainer
    diff = event_sun['final_suggested_price'] - normal_sun['final_suggested_price']
    print(f"Impact of Event: +${diff} in ADR")
