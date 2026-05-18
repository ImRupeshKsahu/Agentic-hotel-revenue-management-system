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
        if hasattr(self, "_original_resolve_api_key"):
            pricing_agent._resolve_api_key = self._original_resolve_api_key
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
        self.assertIn("Candidate optimization", [component["driver"] for component in breakdown["components"]])
        self.assertEqual(price, breakdown["selected_candidate"]["price"])
        self.assertTrue(breakdown["optimizer_candidates"])
        self.assertGreater(price, BASE_PRICE)

    def test_dynamic_pricing_policy_uses_recent_history_and_live_comp_set(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.9178671921,
            day_name="Friday",
            target_date="2017-09-01",
            market_context={
                "comp_low": 130.40,
                "comp_median": 138.99,
                "comp_high": 148.68,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
                "as_of_timestamp": "2017-08-31T00:00:00",
            },
            raw_otb_occupancy=1.0,
            adjusted_otb_occupancy=0.6446,
            return_breakdown=True,
        )

        self.assertEqual(breakdown["base_price"], 128.73)
        self.assertEqual(breakdown["dynamic_min_price"], 123.88)
        self.assertEqual(breakdown["dynamic_max_price"], 166.52)
        self.assertEqual(breakdown["pricing_policy"]["recent_median_adr"], 130.0)
        self.assertEqual(breakdown["pricing_policy"]["same_weekday_median_adr"], 128.18)
        self.assertTrue(breakdown["pricing_policy"]["has_observed_comp_set"])
        self.assertGreaterEqual(price, 138.99)

    def test_ai_suggestion_cannot_change_optimizer_price(self):
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

        self.assertEqual(result["final_adr"], result["optimized_price"])
        self.assertLess(result["final_adr"], 400.0)
        self.assertEqual(result["strategy_applied"], "Accept Optimizer Price")
        self.assertEqual(
            [row["Driver"] for row in result["price_path_components"]],
            [
                "Base rate",
                "Reference price",
                "Candidate optimization",
                "Final recommendation",
            ],
        )
        self.assertEqual(
            [row["Signal"] for row in result["decision_context_components"]],
            [
                "Current booked occupancy",
                "Retained OTB after cancellations",
                "Gross booked pace",
                "Recent pickup trend",
                "Demand anchor",
                "Competitor signal",
                "Market premium headroom",
                "AI advisory",
            ],
        )

    def test_streamlit_metric_frames_reference_as_comparison(self):
        app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")

        self.assertIn('c3.metric("ADR vs Reference"', app_source)
        self.assertNotIn('c3.metric("Reference Delta"', app_source)

    def test_missing_ai_key_keeps_optimizer_price_with_clean_advisory(self):
        self._original_resolve_api_key = pricing_agent._resolve_api_key
        pricing_agent._resolve_api_key = lambda: ""
        pricing_agent.client = None

        result = pricing_agent.run_agentic_pricing(
            target_date="2017-09-02",
            current_occupancy=0.80,
            forecasted_occupancy=0.82,
            shock=0.0,
            competitor_price=130.0,
            booking_velocity=1.0,
        )

        self.assertEqual(result["final_adr"], result["optimized_price"])
        self.assertEqual(result["strategy_applied"], "Review Before Publishing")
        self.assertIn("AI advisory is unavailable", result["strategic_reasoning"])

    def test_ai_owner_summary_is_manager_friendly(self):
        self._original_resolve_api_key = pricing_agent._resolve_api_key
        pricing_agent._resolve_api_key = lambda: "test-key"
        pricing_agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions({
                    "ai_recommended_action": "Review Before Publishing",
                    "ai_risk_level": "Medium",
                    "perceived_demand_strength": "High",
                    "ai_review_flags": [
                        "Optimizer diagnostics show high elasticity against the blended reference price."
                    ],
                    "ai_owner_summary": (
                        "The candidate optimizer selected the ADR from expected revenue among candidates. "
                        "The blended reference price and elasticity diagnostics support the recommendation."
                    ),
                })
            )
        )

        result = pricing_agent.run_agentic_pricing(
            target_date="2017-09-02",
            current_occupancy=0.949,
            forecasted_occupancy=0.93,
            shock=0.0,
            competitor_price=138.41,
            booking_velocity=1.32,
        )

        banned_terms = [
            "blended reference price",
            "candidate optimizer",
            "expected revenue among candidates",
            "elasticity",
            "diagnostics",
        ]
        summary = result["strategic_reasoning"].lower()
        self.assertEqual(result["final_adr"], result["optimized_price"])
        self.assertLessEqual(pricing_agent._sentence_count(result["strategic_reasoning"]), 3)
        self.assertFalse(any(term in summary for term in banned_terms))
        self.assertIn("ADR", result["strategic_reasoning"])
        self.assertIn("Review Before Publishing", result["strategy_applied"])

    def test_manual_shock_changes_optimizer_demand_anchor(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.90,
            pre_shock_occupancy=0.80,
            day_name="Weekday",
            competitor_price=200,
            return_breakdown=True,
        )

        self.assertEqual(breakdown["manual_shock"], 0.1)
        self.assertEqual(breakdown["demand_anchor"], 0.9)
        self.assertEqual(price, breakdown["selected_candidate"]["price"])

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

    def test_recent_pickup_can_strengthen_event_signal(self):
        estimate = estimate_local_intel_impact(
            "100-person wedding block nearby",
            current_occ=0.60,
            forecast_occ=0.75,
            booking_velocity=1.0,
            retained_pace_index=1.0,
            pickup_trend_index=1.25,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["suggested_shock"], 0.10)

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

        self.assertEqual(context_only["rule_based_price"], context_only["optimized_price"])
        self.assertGreater(applied["optimized_price"], context_only["optimized_price"])
        self.assertIn("Local intel was considered as context only", context_only["strategic_reasoning"])

    def test_manager_summary_distinguishes_current_booked_from_forecast(self):
        summary = pricing_agent._manager_summary(
            {
                "optimized_price": 138.41,
                "raw_otb_occupancy": 1.0,
                "adjusted_otb_occupancy": 0.6504,
                "current_occupancy": 0.6504,
                "forecasted_occupancy": 0.932,
                "booking_velocity": 1.0,
                "competitor_price": 138.41,
            },
            "Review Before Publishing",
        )

        self.assertIn("currently 100.0% booked", summary)
        self.assertIn("retained OTB is 65.0%", summary)
        self.assertIn("forecast is 93.2%", summary)
        self.assertNotIn("already 93.2% booked", summary)

    def test_sep_1_regression_uses_sold_out_floor_and_clear_summary(self):
        self._original_resolve_api_key = pricing_agent._resolve_api_key
        pricing_agent._resolve_api_key = lambda: "test-key"
        pricing_agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions({
                    "ai_recommended_action": "Review Before Publishing",
                    "ai_risk_level": "Medium",
                    "perceived_demand_strength": "High",
                    "ai_review_flags": [],
                    "ai_owner_summary": "The recommended ADR is sound because the hotel is already 93.2% booked.",
                })
            )
        )

        result = pricing_agent.run_agentic_pricing(
            target_date="2017-09-01",
            current_occupancy=0.6504,
            forecasted_occupancy=0.9320299768,
            shock=0.0,
            competitor_price=138.41,
            booking_velocity=1.0,
            raw_otb_occupancy=1.0,
            adjusted_otb_occupancy=0.6504,
            expected_cancellations=82.86,
        )

        self.assertGreaterEqual(result["final_adr"], 138.41)
        self.assertTrue(result["optimizer_diagnostics"]["sold_out"])
        self.assertTrue(result["optimizer_diagnostics"]["sold_out_floor_applied"])
        self.assertIn("currently 100.0% booked", result["strategic_reasoning"])
        self.assertIn("retained OTB is 65.0%", result["strategic_reasoning"])
        self.assertIn("forecast is 93.2%", result["strategic_reasoning"])


if __name__ == "__main__":
    unittest.main()
