import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pricing_agent
from config import BASE_PRICE
from local_intel_estimator import estimate_local_intel_impact
from pricing_engine import calculate_recommended_price


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.payload))
                )
            ]
        )


class ExplainablePricingTests(unittest.TestCase):
    def tearDown(self):
        pricing_agent.create_pricing_agent.cache_clear()

    def test_breakdown_components_sum_to_final_price(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.90,
            pre_shock_occupancy=0.80,
            day_name="Saturday",
            competitor_price=130,
            return_breakdown=True,
        )

        total_from_components = round(
            sum(component["adjustment"] for component in breakdown["components"]),
            2,
        )

        self.assertEqual(price, total_from_components)
        self.assertIn("Manual demand adjustment", [component["driver"] for component in breakdown["components"]])
        self.assertGreater(price, BASE_PRICE)

    def test_ai_suggestion_is_clamped_by_guardrails(self):
        pricing_agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions({
                    "baseline_price": 156.0,
                    "final_price": 400.0,
                    "absolute_delta": 244.0,
                    "pct_delta_from_baseline": 156.41,
                    "competitor_gap_pct": 207.69,
                    "strategy_applied": "Inventory Protection",
                    "perceived_demand_strength": "High",
                    "adjustment_components": [],
                    "guardrails_applied": ["No additional guardrail clamp needed"],
                    "owner_summary": "Demand is strong, but guardrails limit the unattended move.",
                })
            )
        )

        result = pricing_agent.run_agentic_pricing(
            target_date="2017-09-02",
            current_occupancy=0.96,
            forecasted_occupancy=0.97,
            shock=0.0,
            competitor_price=130.0,
            booking_velocity=1.4,
        )

        self.assertEqual(result["final_adr"], result["adjustment_band"]["max_allowed"])
        self.assertLess(result["final_adr"], 400.0)
        self.assertTrue(any("clamped" in item for item in result["guardrails_applied"]))
        self.assertEqual(
            [row["Driver"] for row in result["price_components"]],
            [
                "Base rate",
                "Occupancy lift",
                "Manual demand adjustment",
                "Local intel adjustment",
                "Day-of-week effect",
                "Competitor cap",
                "Pace premium",
                "Final recommendation",
            ],
        )

    def test_manual_shock_is_explained_once_in_baseline(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.90,
            pre_shock_occupancy=0.80,
            day_name="Weekday",
            competitor_price=200,
            return_breakdown=True,
        )
        event_component = next(
            component for component in breakdown["components"]
            if component["driver"] == "Manual demand adjustment"
        )

        self.assertEqual(event_component["adjustment"], 10.0)
        self.assertEqual(price, 140.0)

    def test_traffic_jam_defaults_to_context_only(self):
        estimate = estimate_local_intel_impact(
            "big traffic jam at the entry of the city",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Operational Disruption / Ambiguous Demand")
        self.assertEqual(estimate["suggested_shock"], 0.0)
        self.assertFalse(estimate["apply_allowed"])

    def test_wedding_block_suggests_but_does_not_auto_apply(self):
        estimate = estimate_local_intel_impact(
            "100-person wedding block nearby",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertGreater(estimate["suggested_shock"], 0.0)
        self.assertTrue(estimate["apply_allowed"])

    def test_major_conference_sold_out_is_capped_positive(self):
        estimate = estimate_local_intel_impact(
            "major conference nearby and nearby hotels are sold out",
            current_occ=0.80,
            forecast_occ=0.92,
            booking_velocity=1.3,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["confidence"], "High")
        self.assertEqual(estimate["suggested_shock"], 0.15)
        self.assertTrue(estimate["apply_allowed"])

    def test_fifa_world_cup_final_is_event(self):
        estimate = estimate_local_intel_impact(
            "FIFA world cup final on that day",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["confidence"], "Medium")
        self.assertEqual(estimate["suggested_shock"], 0.05)
        self.assertTrue(estimate["apply_allowed"])

    def test_fifa_world_cup_sold_out_is_high_confidence_event(self):
        estimate = estimate_local_intel_impact(
            "FIFA world cup final sold out nearby hotels",
            current_occ=0.80,
            forecast_occ=0.90,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["confidence"], "High")
        self.assertEqual(estimate["suggested_shock"], 0.15)
        self.assertTrue(estimate["apply_allowed"])

    def test_nearby_hotels_sold_out_due_to_fifa_final_is_not_competitor_only(self):
        estimate = estimate_local_intel_impact(
            "nearby hotels sold out due to FIFA final",
            current_occ=0.80,
            forecast_occ=0.90,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["confidence"], "High")
        self.assertGreater(estimate["suggested_shock"], 0.0)

    def test_airport_shutdown_is_bounded_disruption(self):
        estimate = estimate_local_intel_impact(
            "airport shutdown and road closure near the city",
            current_occ=0.80,
            forecast_occ=0.80,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Operational Disruption / Ambiguous Demand")
        self.assertLessEqual(abs(estimate["suggested_shock"]), 0.10)
        self.assertTrue(any("Disruption" in item or "disruption" in item for item in estimate["guardrails_applied"]))

    def test_local_intel_changes_baseline_only_when_applied(self):
        pricing_agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions({
                    "baseline_price": 120.0,
                    "final_price": 120.0,
                    "absolute_delta": 0.0,
                    "pct_delta_from_baseline": 0.0,
                    "competitor_gap_pct": 0.0,
                    "strategy_applied": "Baseline Hold",
                    "perceived_demand_strength": "Medium",
                    "adjustment_components": [],
                    "guardrails_applied": ["No additional guardrail clamp needed"],
                    "owner_summary": "Baseline held after local intel review.",
                })
            )
        )
        estimate = estimate_local_intel_impact(
            "100-person wedding block nearby",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )

        context_only = pricing_agent.run_agentic_pricing(
            target_date="2017-09-04",
            current_occupancy=0.50,
            forecasted_occupancy=0.70,
            shock=0.0,
            competitor_price=200.0,
            booking_velocity=1.0,
            manual_event_text="100-person wedding block nearby",
            local_intel_estimate=estimate,
            local_intel_applied_shock=0.0,
        )
        applied = pricing_agent.run_agentic_pricing(
            target_date="2017-09-04",
            current_occupancy=0.50,
            forecasted_occupancy=0.70,
            shock=0.0,
            competitor_price=200.0,
            booking_velocity=1.0,
            manual_event_text="100-person wedding block nearby",
            local_intel_estimate=estimate,
            local_intel_applied_shock=estimate["suggested_shock"],
        )

        self.assertEqual(context_only["rule_based_price"], 120.0)
        self.assertGreater(applied["rule_based_price"], context_only["rule_based_price"])
        self.assertIn("Local intel was considered as context only", context_only["strategic_reasoning"])


if __name__ == "__main__":
    unittest.main()
