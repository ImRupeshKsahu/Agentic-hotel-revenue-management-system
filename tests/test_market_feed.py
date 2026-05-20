import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_feed import build_simulated_market_snapshot, simulate_competitor_market_event
from pms_snapshot import export_live_market_state
from pricing_agent import run_agentic_pricing
from pricing_engine import calculate_recommended_price


class MarketFeedTests(unittest.TestCase):
    def test_simulator_emits_future_ready_schema_and_distinct_regimes(self):
        snapshot = build_simulated_market_snapshot(
            pd.date_range("2026-06-01", periods=7, freq="D"),
            as_of_timestamp="2026-05-16",
            seed=7,
        )

        self.assertEqual(
            list(snapshot.columns),
            [
                "stay_date",
                "as_of_timestamp",
                "comp_low",
                "comp_median",
                "comp_high",
                "sample_size",
                "source_quality",
                "market_regime",
            ],
        )
        self.assertGreaterEqual(snapshot["market_regime"].nunique(), 4)
        self.assertTrue((snapshot["comp_low"] <= snapshot["comp_median"]).all())
        self.assertTrue((snapshot["comp_median"] <= snapshot["comp_high"]).all())

    def test_simulated_live_market_event_changes_a_regime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "market.csv")
            event = simulate_competitor_market_event(
                stay_dates=pd.date_range("2026-06-01", periods=5, freq="D"),
                as_of_timestamp="2026-05-16",
                path=path,
                seed=11,
            )

            self.assertNotEqual(event["old_regime"], event["new_regime"])
            self.assertTrue(Path(path).exists())

    def test_live_state_consumes_comp_set_snapshot(self):
        otb = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-01"]),
                "Live_OTB": [100],
                "Adjusted_OTB": [95.0],
                "Expected_Cancellations": [5.0],
                "OTB_Occupancy": [100 / 237],
                "Adjusted_OTB_Occupancy": [95 / 237],
                "OTB_ADR": [120.0],
                "Capacity": [237],
                "Historical_Avg_OTB": [80],
                "Booking_Velocity": [1.25],
                "Gross_Pace_Index": [1.25],
                "Retained_Pace_Index": [1.15],
                "Pickup_Trend_Index": [1.30],
                "Pricing_Pace_Index": [1.20],
            }
        )
        market = pd.DataFrame(
            [
                {
                    "stay_date": "2026-06-01",
                    "as_of_timestamp": "2026-05-16T00:00:00",
                    "comp_low": 120.0,
                    "comp_median": 135.0,
                    "comp_high": 150.0,
                    "sample_size": 5,
                    "source_quality": "simulated",
                    "market_regime": "event_compression",
                }
            ]
        )

        state = export_live_market_state(otb, market_snapshots=market)
        day = state["2026-06-01"]

        self.assertEqual(day["booked_adr"], 120.0)
        self.assertEqual(day["competitor_price"], 135.0)
        self.assertEqual(day["comp_low"], 120.0)
        self.assertEqual(day["comp_high"], 150.0)
        self.assertEqual(day["market_regime"], "event_compression")
        self.assertEqual(day["gross_pace_index"], 1.25)
        self.assertEqual(day["retained_pace_index"], 1.15)
        self.assertEqual(day["pickup_trend_index"], 1.30)
        self.assertEqual(day["pricing_pace_index"], 1.20)

    def test_compression_allows_more_premium_than_soft_market(self):
        market_context = {
            "comp_low": 120.0,
            "comp_median": 130.0,
            "comp_high": 145.0,
            "sample_size": 5,
            "source_quality": "simulated",
            "market_regime": "normal_market",
            "as_of_timestamp": "2026-05-16T00:00:00",
        }
        _, _, soft = calculate_recommended_price(
            occupancy=0.50,
            day_name="Monday",
            target_date="2026-06-15",
            market_context=market_context,
            booking_velocity=0.75,
            raw_otb_occupancy=0.50,
            adjusted_otb_occupancy=0.48,
            return_breakdown=True,
        )
        compressed_price, _, compressed = calculate_recommended_price(
            occupancy=0.97,
            day_name="Monday",
            target_date="2026-05-19",
            market_context={**market_context, "market_regime": "market_wide_sellout"},
            booking_velocity=1.35,
            raw_otb_occupancy=0.99,
            adjusted_otb_occupancy=0.96,
            return_breakdown=True,
        )

        self.assertLess(soft["allowed_premium_pct"], compressed["allowed_premium_pct"])
        self.assertLess(soft["compression_score"], compressed["compression_score"])
        self.assertGreater(compressed_price, market_context["comp_median"])

    def test_pricing_uses_composite_pace_signal_when_available(self):
        _, _, legacy = calculate_recommended_price(
            occupancy=0.80,
            day_name="Monday",
            competitor_price=140.0,
            booking_velocity=1.0,
            return_breakdown=True,
        )
        _, _, richer = calculate_recommended_price(
            occupancy=0.80,
            day_name="Monday",
            competitor_price=140.0,
            booking_velocity=1.0,
            gross_pace_index=1.0,
            retained_pace_index=1.20,
            pickup_trend_index=1.25,
            pricing_pace_index=1.18,
            return_breakdown=True,
        )

        self.assertEqual(legacy["pricing_pace_index"], 1.0)
        self.assertEqual(richer["gross_pace_index"], 1.0)
        self.assertEqual(richer["pricing_pace_index"], 1.18)
        self.assertGreater(richer["demand_anchor"], legacy["demand_anchor"])

    def test_legacy_single_competitor_input_still_works(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.82,
            day_name="Friday",
            competitor_price=140.0,
            return_breakdown=True,
        )

        self.assertGreater(price, 0)
        self.assertEqual(breakdown["market_context"]["comp_low"], 140.0)
        self.assertEqual(breakdown["market_context"]["comp_median"], 140.0)
        self.assertEqual(breakdown["market_context"]["comp_high"], 140.0)

    def test_decision_log_context_is_present_in_result(self):
        result = run_agentic_pricing(
            target_date="2026-06-01",
            current_occupancy=0.92,
            forecasted_occupancy=0.95,
            shock=0.0,
            market_context={
                "comp_low": 120.0,
                "comp_median": 135.0,
                "comp_high": 150.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "event_compression",
                "as_of_timestamp": "2026-05-16T00:00:00",
            },
            competitor_price=135.0,
            booking_velocity=1.25,
            record_decision=False,
        )

        self.assertIn("market_context", result)
        self.assertIn("compression_score", result["optimizer_diagnostics"])
        self.assertIn("allowed_premium_pct", result["optimizer_diagnostics"])


if __name__ == "__main__":
    unittest.main()
