import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import manager_copilot


class ManagerCopilotTests(unittest.TestCase):
    def _forecast_df(self):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2017-09-01", "2017-09-02"]),
                "Forecasted_Occupancy": [0.82, 0.78],
                "Competitor_Rate": [140.0, 130.0],
            }
        )

    def _live_data(self):
        return {
            "2017-09-01": {
                "raw_otb_occupancy": 1.0,
                "adjusted_otb_occupancy": 0.65,
                "expected_cancellations": 82.0,
                "booked_adr": 129.84,
                "competitor_price": 140.0,
                "comp_low": 135.0,
                "comp_median": 140.0,
                "comp_high": 150.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
                "pickup_trend_index": 1.30,
                "gross_pace_index": 1.10,
                "retained_pace_index": 1.05,
                "pricing_pace_index": 1.08,
                "booking_velocity": 1.10,
                "total_rooms": 237,
            },
            "2017-09-02": {
                "raw_otb_occupancy": 0.70,
                "adjusted_otb_occupancy": 0.68,
                "expected_cancellations": 4.0,
                "booked_adr": 145.0,
                "competitor_price": 130.0,
                "comp_low": 125.0,
                "comp_median": 130.0,
                "comp_high": 138.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
                "pickup_trend_index": 1.00,
                "gross_pace_index": 1.00,
                "retained_pace_index": 1.00,
                "pricing_pace_index": 1.00,
                "booking_velocity": 1.00,
                "total_rooms": 237,
            },
        }

    def test_rank_top_opportunities_uses_revenue_upside(self):
        records = [
            {"date": "2017-09-01", "revenue_upside": 120.0, "expected_revenue": 1000.0},
            {"date": "2017-09-02", "revenue_upside": 350.0, "expected_revenue": 900.0},
        ]
        ranked = manager_copilot.rank_top_opportunities(records)
        self.assertEqual([row["date"] for row in ranked], ["2017-09-02", "2017-09-01"])

    def test_risk_dates_surface_even_when_upside_is_low(self):
        records = [
            {
                "date": "2017-09-01",
                "review_status": "Review needed",
                "manual_approval_required": True,
                "sold_out": True,
                "material_retention_gap": True,
                "review_flags": ["Review before publishing."],
                "revenue_upside": 0.0,
            },
            {
                "date": "2017-09-02",
                "review_status": "No review",
                "manual_approval_required": False,
                "sold_out": False,
                "material_retention_gap": False,
                "review_flags": [],
                "revenue_upside": 500.0,
            },
        ]
        ranked = manager_copilot.rank_top_risks(records)
        self.assertEqual([row["date"] for row in ranked], ["2017-09-01"])

    def test_sold_out_retention_gap_is_flagged_and_occupancy_fields_stay_distinct(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        sold_out = next(row for row in records if row["date"] == "2017-09-01")

        self.assertTrue(sold_out["sold_out"])
        self.assertTrue(sold_out["material_retention_gap"])
        self.assertEqual(sold_out["review_status"], "Review needed")
        self.assertTrue(any("fully booked on paper" in flag for flag in sold_out["review_flags"]))
        self.assertEqual(sold_out["raw_otb_occupancy"], 1.0)
        self.assertEqual(sold_out["adjusted_otb_occupancy"], 0.65)
        self.assertEqual(sold_out["forecasted_occupancy"], 0.82)
        self.assertEqual(sold_out["booked_adr"], 129.84)
        self.assertEqual(sold_out["expected_cancellations"], 82.0)
        self.assertEqual(sold_out["current_otb"], 237.0)
        self.assertEqual(sold_out["adjusted_otb"], 154.05)
        self.assertTrue(any("fully booked on paper" in reason for reason in sold_out["top_reasons"]))
        self.assertTrue(any("likely retained occupancy" in flag for flag in sold_out["review_flags"]))
        self.assertFalse(any("raw otb" in flag.lower() for flag in sold_out["review_flags"]))

    def test_revenue_upside_is_measured_against_booked_adr_not_reference_price(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        second_date = next(row for row in records if row["date"] == "2017-09-02")

        self.assertEqual(second_date["booked_adr"], 145.0)
        self.assertGreater(second_date["expected_rooms"], 0)
        self.assertGreater(second_date["booked_adr_proxy_expected_rooms"], 0)
        self.assertGreater(second_date["reference_adr"], 0)
        self.assertEqual(
            second_date["revenue_upside"],
            round(
                max(0.0, second_date["expected_revenue"] - second_date["booked_adr_revenue_proxy"]),
                2,
            ),
        )
        self.assertNotEqual(second_date["booked_adr_revenue_proxy"], second_date["reference_revenue_proxy"])

    def test_briefing_fallback_is_manager_friendly_and_does_not_mutate_records(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        before_prices = [row["recommended_adr"] for row in records]
        payload = manager_copilot.build_briefing_payload(records)

        with patch.object(manager_copilot, "_resolve_api_key", return_value=""):
            briefing = manager_copilot.generate_executive_briefing(payload)

        self.assertIn("Focus first on", briefing)
        self.assertNotIn("optimizer", briefing.lower())
        self.assertEqual(before_prices, [row["recommended_adr"] for row in records])

    def test_sanitize_executive_briefing_removes_code_marks_and_currency_spacing(self):
        cleaned = manager_copilot.sanitize_executive_briefing("`Focus on $ 538 upside.`")
        self.assertEqual(cleaned, "Focus on $538 upside.")

    def test_market_outlook_metrics_use_room_nights_and_high_demand_regimes(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        records[0]["market_regime"] = "event_compression"
        records[1]["market_regime"] = "market_wide_sellout"

        metrics = manager_copilot.build_market_outlook_metrics(records)

        self.assertEqual(metrics["booked_room_nights"], 402.9)
        self.assertEqual(metrics["retained_room_nights"], 315.21)
        self.assertEqual(metrics["high_demand_market_dates"], 2)

    def test_champion_model_audit_compares_selection_and_recent_audit(self):
        champion_payload = {
            "model": "random_forest_recursive",
            "metrics": {
                "MAE_pp": 8.71504,
                "RMSE_pp": 10.4321,
                "Bias_pp": 0.35035,
                "Accuracy": 87.1504,
                "WAPE": 12.8496,
                "Bias": 0.0035035,
                "Stability": 0.927529,
            },
            "backtest_metadata": {"audit_status": "ok"},
        }
        audit_summary = pd.DataFrame(
            [
                {
                    "Model": "random_forest_recursive",
                    "MAE_pp": 2.9025,
                    "RMSE_pp": 3.4567,
                    "Bias_pp": -1.37357,
                    "Accuracy": 97.0975,
                    "WAPE": 2.9025,
                    "Bias": -0.0137357,
                    "Stability": 0.972786,
                    "Interval_Coverage": 1.0,
                    "Audit_Status": "ok",
                    "Is_Champion": True,
                }
            ]
        )

        audit = manager_copilot.build_champion_model_audit(champion_payload, audit_summary)

        self.assertEqual(audit["champion_model"], "random_forest_recursive")
        self.assertEqual(audit["recent_avg_occupancy_miss_pp"], 2.9025)
        rows = {row["Metric"]: row for row in audit["rows"]}
        self.assertEqual(rows["Avg Occupancy Miss (MAE)"]["Selection Backtest"], "8.72 pp")
        self.assertEqual(rows["Avg Occupancy Miss (MAE)"]["Recent Audit"], "2.90 pp")
        self.assertEqual(rows["Large Miss Guardrail (RMSE)"]["Selection Backtest"], "10.43 pp")
        self.assertEqual(rows["Large Miss Guardrail (RMSE)"]["Recent Audit"], "3.46 pp")
        self.assertEqual(rows["Bias"]["Selection Backtest"], "+0.35 pp")
        self.assertEqual(rows["Bias"]["Recent Audit"], "-1.37 pp")
        self.assertEqual(rows["Interval Coverage"]["Recent Audit"], "100.0%")
        self.assertEqual(rows["Audit Status"]["Recent Audit"], "Ok")

    def test_overstepping_ai_briefing_falls_back_to_bounded_copy(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        payload = manager_copilot.build_briefing_payload(records)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content='{"executive_briefing": "Raise rates on the strongest dates immediately."}'
                                )
                            )
                        ]
                    )
                )
            )
        )

        with patch.object(manager_copilot, "_resolve_api_key", return_value="test-key"), patch.object(
            manager_copilot, "_get_client", return_value=fake_client
        ):
            briefing = manager_copilot.generate_executive_briefing(payload)

        self.assertIn("Focus first on", briefing)
        self.assertNotIn("raise rates", briefing.lower())

    def test_technical_ai_briefing_falls_back_to_manager_copy(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        payload = manager_copilot.build_briefing_payload(records)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content='{"executive_briefing": "Sep 1 has $538 upside, but raw OTB sellout and retained occupancy require review."}'
                                )
                            )
                        ]
                    )
                )
            )
        )

        with patch.object(manager_copilot, "_resolve_api_key", return_value="test-key"), patch.object(
            manager_copilot, "_get_client", return_value=fake_client
        ):
            briefing = manager_copilot.generate_executive_briefing(payload)

        self.assertIn("Focus first on", briefing)
        self.assertNotIn("raw OTB", briefing)

    def test_manager_facing_views_source_contains_core_sections(self):
        app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
        self.assertIn('["Morning Briefing", "Market Outlook", "Scenario Lab"]', app_source)
        self.assertIn('st.subheader("Executive Briefing")', app_source)
        self.assertIn('c1.metric("Total Revenue Upside"', app_source)
        self.assertIn('c3.metric("Expected Cancellations"', app_source)
        self.assertIn('st.subheader("Top Revenue Opportunities")', app_source)
        self.assertIn('st.subheader("Top Risks / Review Needed")', app_source)
        self.assertIn('st.subheader("30-Day Snapshot")', app_source)
        self.assertIn('st.subheader(f"Date Detail — {selected_date}")', app_source)
        self.assertIn("build_revenue_upside_basis_text(selected_record)", app_source)
        self.assertIn('st.subheader("Champion Model Audit")', app_source)
        self.assertIn('if st.sidebar.button("Run Scenario"', app_source)
        self.assertIn("def render_scenario_result", app_source)
        self.assertIn("render_scenario_result(latest_result", app_source)
        self.assertIn("render_scenario_result(agent_result, scenario_state=current_state)", app_source)
        self.assertIn("render_technical_trace(agent_result, use_expander=technical_expander)", app_source)
        self.assertIn("render_price_trace(agent_result)", app_source)
        self.assertIn('"Likely Retained": format_pct', app_source)
        self.assertIn('"Booked Rooms": f"', app_source)
        self.assertIn('"Recommended ADR": f"', app_source)
        self.assertIn('"Stayover Rooms"', app_source)
        self.assertIn('"Future Arrival Rooms"', app_source)
        self.assertIn("in-house stayovers are treated as retained", app_source)
        self.assertIn('st.subheader("Scenario Copilot Chat")', app_source)
        self.assertIn('st.chat_input("Ask Scenario Copilot")', app_source)
        self.assertIn("handle_grounded_scenario_chat", app_source)
        self.assertIn("if response.confirmation_prompt:", app_source)
        self.assertIn("scenario_copilot_clarification_count", app_source)
        self.assertIn("scenario_copilot_memory", app_source)
        self.assertIn("update_conversation_memory", app_source)
        self.assertIn("scenario_horizon_records = build_opportunity_records", app_source)
        self.assertIn("horizon_records=scenario_horizon_records", app_source)
        self.assertIn("def reset_scenario_copilot_chat", app_source)
        self.assertIn('st.button("Start Over"', app_source)
        self.assertIn("scenario_copilot_pending_draft = None", app_source)
        self.assertIn("scenario_copilot_latest_result = None", app_source)
        self.assertIn("scenario_copilot_clarification_count = 0", app_source)
        self.assertIn("scenario_copilot_memory = ScenarioConversationMemory()", app_source)


if __name__ == "__main__":
    unittest.main()
