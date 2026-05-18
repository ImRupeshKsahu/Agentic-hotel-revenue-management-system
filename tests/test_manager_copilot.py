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
        self.assertTrue(any("Sold-out OTB" in flag for flag in sold_out["review_flags"]))
        self.assertEqual(sold_out["raw_otb_occupancy"], 1.0)
        self.assertEqual(sold_out["adjusted_otb_occupancy"], 0.65)
        self.assertEqual(sold_out["forecasted_occupancy"], 0.82)
        self.assertEqual(sold_out["booked_adr"], 129.84)

    def test_revenue_upside_is_measured_against_booked_adr_not_reference_price(self):
        records = manager_copilot.build_opportunity_records(self._forecast_df(), self._live_data())
        second_date = next(row for row in records if row["date"] == "2017-09-02")

        self.assertEqual(second_date["booked_adr"], 145.0)
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

    def test_today_view_source_contains_core_sections(self):
        app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
        self.assertIn('["Today", "📈 Market Performance", "🤖 Agentic Simulation"]', app_source)
        self.assertIn('st.subheader("Executive Briefing")', app_source)
        self.assertIn('st.subheader("Top Revenue Opportunities")', app_source)
        self.assertIn('st.subheader("Top Risks / Review Needed")', app_source)
        self.assertIn('st.subheader("30-Day Snapshot")', app_source)
        self.assertIn('st.subheader(f"Date Detail — {selected_date}")', app_source)


if __name__ == "__main__":
    unittest.main()
