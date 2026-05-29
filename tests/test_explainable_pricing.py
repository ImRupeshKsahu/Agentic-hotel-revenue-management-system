import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import copilot_core.pricing_agent as pricing_agent
from project_core.config import BASE_PRICE
from pricing_core.local_intel import (
    estimate_local_intel_impact,
    infer_local_intel_proximity,
    local_intel_events_for_date,
)
from pricing_core.engine import calculate_recommended_price
from utils.utility_functions import escape_streamlit_markdown


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
                "Likely retained occupancy",
                "Booked pace",
                "Recent pickup trend",
                "Demand used for pricing",
                "Competitor median",
                "Allowed premium vs market",
                "AI advisory",
            ],
        )

    def test_streamlit_metric_frames_reference_as_comparison(self):
        app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")

        self.assertIn('metric("ADR vs Reference"', app_source)
        self.assertNotIn('metric("Reference Delta"', app_source)

    def test_streamlit_currency_is_escaped_before_markdown_rendering(self):
        self.assertEqual(escape_streamlit_markdown("Upside: $538"), r"Upside: \$538")

    def test_manager_copy_surfaces_use_markdown_safe_rendering(self):
        app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
        self.assertEqual(app_source.count("escape_streamlit_markdown(normalize_reasoning("), 3)
        self.assertIn("Current booked rooms:", app_source)
        self.assertIn("Comp set (low / median / high):", app_source)

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
        self.assertEqual(estimate["proximity_bucket"], "nearby")

    def test_manual_proximity_infers_similar_words(self):
        nearby = infer_local_intel_proximity("concert next door to the hotel")
        city_center = infer_local_intel_proximity("festival in the city centre")
        citywide = infer_local_intel_proximity("conference across the city with hotels full")
        distant = infer_local_intel_proximity("event outside Lisbon in the outskirts")

        self.assertEqual(nearby["bucket"], "nearby")
        self.assertEqual(city_center["bucket"], "city_center_relevant")
        self.assertEqual(citywide["bucket"], "citywide")
        self.assertEqual(distant["bucket"], "distant_or_uncertain")

    def test_manual_proximity_changes_manual_event_impact(self):
        unspecified = estimate_local_intel_impact(
            "music festival",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )
        nearby = estimate_local_intel_impact(
            "music festival nearby",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )
        distant_override = estimate_local_intel_impact(
            "music festival nearby",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
            manual_proximity_bucket="distant_or_uncertain",
        )

        self.assertEqual(unspecified["proximity_bucket"], "not_specified")
        self.assertEqual(nearby["proximity_bucket"], "nearby")
        self.assertEqual(distant_override["proximity_source"], "manager_override")
        self.assertGreater(nearby["suggested_shock"], unspecified["suggested_shock"])
        self.assertLess(distant_override["suggested_shock"], nearby["suggested_shock"])

    def test_major_conference_sold_out_is_capped_positive(self):
        estimate = estimate_local_intel_impact(
            "major conference nearby and nearby hotels are sold out",
            current_occ=0.80,
            forecast_occ=0.92,
            booking_velocity=1.3,
        )

        self.assertEqual(estimate["classification"], "Event")
        self.assertEqual(estimate["confidence"], "High")
        self.assertEqual(estimate["suggested_shock"], 0.08)
        self.assertGreater(estimate["adr_headroom"], 0.0)
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
        self.assertEqual(estimate["suggested_shock"], 0.12)

    def test_fifa_world_cup_final_is_event(self):
        estimate = estimate_local_intel_impact(
            "FIFA world cup final on that day",
            current_occ=0.50,
            forecast_occ=0.70,
            booking_velocity=1.0,
        )

        self.assertEqual(estimate["classification"], "Major Sports Event")
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
        self.assertEqual(estimate["suggested_shock"], 0.10)
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

    def test_lisbon_seeded_events_load_for_september_horizon(self):
        events = local_intel_events_for_date("2017-09-12")

        self.assertTrue(events)
        self.assertEqual(events[0]["event_name"], "Benfica vs CSKA Moscow")
        self.assertTrue(events[0]["source_url"])

    def test_dates_without_seeded_events_are_context_free(self):
        self.assertEqual(local_intel_events_for_date("2017-09-30"), [])

    def test_benfica_match_scores_stronger_than_small_business_event(self):
        match = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.50,
            forecast_occ=0.70,
            target_date="2017-09-12",
        )
        summit = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.50,
            forecast_occ=0.70,
            target_date="2017-09-26",
        )

        self.assertEqual(match["source"], "seeded_lisbon_calendar")
        self.assertEqual(match["classification"], "Major Sports Event")
        self.assertGreater(match["suggested_shock"], summit["suggested_shock"])
        self.assertGreater(match["adr_headroom"], summit["adr_headroom"])

    def test_manual_event_combines_with_seeded_calendar_event_as_cluster(self):
        calendar_only = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.50,
            forecast_occ=0.70,
            target_date="2017-09-12",
        )
        cluster = estimate_local_intel_impact(
            "music event nearby",
            current_occ=0.50,
            forecast_occ=0.70,
            target_date="2017-09-12",
        )

        self.assertEqual(cluster["classification"], "Local Event Cluster")
        self.assertEqual(cluster["source"], "seeded_calendar_plus_manual_intel")
        self.assertGreater(cluster["suggested_shock"], calendar_only["suggested_shock"])
        self.assertGreaterEqual(cluster["adr_headroom"], calendar_only["adr_headroom"])
        self.assertTrue(any("combined as an event cluster" in item for item in cluster["guardrails_applied"]))
        self.assertTrue(any("Benfica" in item for item in cluster["evidence"]))
        self.assertTrue(any("music event nearby" in item for item in cluster["evidence"]))

    def test_lisbon_cultural_cluster_has_medium_city_center_impact(self):
        estimate = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.50,
            forecast_occ=0.70,
            target_date="2017-09-15",
        )

        self.assertIn(estimate["classification"], {"Cultural / Festival Cluster", "Local Event Cluster"})
        self.assertEqual(estimate["confidence"], "Medium")
        self.assertGreaterEqual(estimate["suggested_shock"], 0.04)
        self.assertGreater(estimate["adr_headroom"], 0.0)

    def test_seeded_event_shock_clamps_when_forecast_is_near_full(self):
        estimate = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.90,
            forecast_occ=0.95,
            target_date="2017-09-12",
        )

        self.assertEqual(estimate["suggested_shock"], 0.05)
        self.assertTrue(any("100%" in item for item in estimate["guardrails_applied"]))

    def test_seeded_business_event_preserves_raw_demand_when_clamped(self):
        estimate = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=1.00,
            forecast_occ=0.95,
            target_date="2017-09-26",
        )

        self.assertEqual(estimate["classification"], "Business Event")
        self.assertEqual(estimate["suggested_shock"], 0.0)
        self.assertGreater(estimate["raw_occupancy_shock"], 0.0)
        self.assertAlmostEqual(estimate["raw_occupancy_shock_pct"], 2.7)
        self.assertTrue(estimate["demand_was_clamped"])
        self.assertIn("occupancy room", estimate["demand_clamp_reason"])
        self.assertGreater(estimate["adr_headroom"], 0.0)

    def test_approved_local_intel_headroom_changes_pricing_policy(self):
        estimate = estimate_local_intel_impact(
            "local intel event overlay",
            current_occ=0.70,
            forecast_occ=0.75,
            target_date="2017-09-12",
            market_context={
                "comp_low": 130.0,
                "comp_median": 150.0,
                "comp_high": 170.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
            },
        )

        _, _, baseline = calculate_recommended_price(
            occupancy=0.75,
            day_name="Tuesday",
            target_date="2017-09-12",
            competitor_price=150.0,
            market_context={
                "comp_low": 130.0,
                "comp_median": 150.0,
                "comp_high": 170.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
            },
            return_breakdown=True,
        )
        _, _, applied = calculate_recommended_price(
            occupancy=0.85,
            day_name="Tuesday",
            target_date="2017-09-12",
            competitor_price=150.0,
            market_context={
                "comp_low": 130.0,
                "comp_median": 150.0,
                "comp_high": 170.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
            },
            return_breakdown=True,
            pre_shock_occupancy=0.75,
            local_intel_shock=estimate["suggested_shock"],
            local_intel_adr_headroom_pct=estimate["adr_headroom"],
            manual_event_text="local intel event overlay",
        )

        self.assertGreaterEqual(applied["allowed_premium_pct"], estimate["adr_headroom"])
        self.assertGreater(applied["allowed_premium_pct"], baseline["allowed_premium_pct"])

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
        self.assertIn("likely retained occupancy is 65.0%", summary)
        self.assertIn("forecast occupancy is 93.2%", summary)
        self.assertNotIn("already 93.2% booked", summary)

    def test_manager_summary_mentions_applied_local_intel_overlay(self):
        summary = pricing_agent._manager_summary(
            {
                "optimized_price": 135.0,
                "raw_otb_occupancy": 0.717,
                "adjusted_otb_occupancy": 0.664,
                "current_occupancy": 0.664,
                "forecasted_occupancy": 0.917,
                "gross_pace_index": 1.0,
                "pickup_trend_index": 1.25,
                "competitor_price": 135.67,
                "market_context": {"comp_median": 135.67},
                "manual_event_text": "music event nearby",
                "local_intel_estimate": {
                    "classification": "Local Event Cluster",
                    "suggested_shock": 0.08,
                    "adr_headroom": 0.12,
                },
                "local_intel_applied_shock": 0.08,
                "local_intel_applied_adr_headroom": 0.12,
            },
            "Review Before Publishing",
        )

        self.assertIn("local intel is included", summary)
        self.assertIn("+8.0% demand", summary)
        self.assertIn("+12.0% ADR headroom", summary)

    def test_manager_summary_mentions_context_only_local_intel(self):
        summary = pricing_agent._manager_summary(
            {
                "optimized_price": 135.0,
                "raw_otb_occupancy": 0.717,
                "adjusted_otb_occupancy": 0.664,
                "current_occupancy": 0.664,
                "forecasted_occupancy": 0.917,
                "gross_pace_index": 1.0,
                "pickup_trend_index": 1.25,
                "competitor_price": 135.67,
                "market_context": {"comp_median": 135.67},
                "manual_event_text": "music event nearby",
                "local_intel_estimate": {
                    "classification": "Event",
                    "suggested_shock": 0.05,
                    "adr_headroom": 0.04,
                },
                "local_intel_applied_shock": 0.0,
                "local_intel_applied_adr_headroom": 0.0,
            },
            "Review Before Publishing",
        )

        self.assertIn("local intel was reviewed as context only", summary)
        self.assertIn("not included in priced demand", summary)

    def test_manager_summary_omits_local_intel_when_none_supplied(self):
        summary = pricing_agent._manager_summary(
            {
                "optimized_price": 135.0,
                "raw_otb_occupancy": 0.717,
                "adjusted_otb_occupancy": 0.664,
                "current_occupancy": 0.664,
                "forecasted_occupancy": 0.917,
                "gross_pace_index": 1.0,
                "pickup_trend_index": 1.25,
                "competitor_price": 135.67,
                "market_context": {"comp_median": 135.67},
                "manual_event_text": "",
                "local_intel_estimate": {},
                "local_intel_applied_shock": 0.0,
                "local_intel_applied_adr_headroom": 0.0,
            },
            "Review Before Publishing",
        )

        self.assertNotIn("local intel", summary.lower())
        self.assertNotIn("context only", summary.lower())

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
        self.assertIn("likely retained occupancy is 65.0%", result["strategic_reasoning"])
        self.assertIn("forecast occupancy is 93.2%", result["strategic_reasoning"])


if __name__ == "__main__":
    unittest.main()
