import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancellation_risk import estimate_cancellation_probabilities
from pms_snapshot import calculate_otb_snapshot
from pricing_agent import optimizer_node
from pricing_engine import calculate_recommended_price


def booking(
    booking_id,
    arrival_date,
    booking_date,
    *,
    is_canceled=0,
    cancellation_date=None,
    lead_time=10,
    market_segment="Direct",
    distribution_channel="Direct",
    customer_type="Transient",
):
    arrival_date = pd.Timestamp(arrival_date)
    return {
        "booking_id": booking_id,
        "arrival_date": arrival_date,
        "departure_date": arrival_date + pd.Timedelta(days=1),
        "booking_date": pd.Timestamp(booking_date),
        "stay_nights": 1,
        "reservation_status": "Canceled" if is_canceled else "Check-Out",
        "reservation_status_date": pd.Timestamp(cancellation_date) if cancellation_date else arrival_date + pd.Timedelta(days=1),
        "is_canceled": is_canceled,
        "cancellation_date": pd.Timestamp(cancellation_date) if cancellation_date else pd.NaT,
        "adr": 120.0,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "customer_type": customer_type,
        "lead_time": lead_time,
    }


class CancellationAdjustedOTBTests(unittest.TestCase):
    def test_specific_bucket_rate_is_used_when_supported(self):
        as_of = pd.Timestamp("2024-01-31")
        history = pd.DataFrame(
            [
                booking(1, "2024-01-10", "2023-12-20", is_canceled=1, cancellation_date="2024-01-01"),
                booking(2, "2024-01-11", "2023-12-20", is_canceled=1, cancellation_date="2024-01-02"),
                booking(3, "2024-01-12", "2023-12-20", is_canceled=0),
            ]
        )
        active = pd.DataFrame([booking(4, "2024-02-10", "2024-01-20")])

        result = estimate_cancellation_probabilities(
            active,
            history,
            as_of,
            min_support=3,
            smoothing=0,
        )

        self.assertAlmostEqual(result.iloc[0]["cancellation_probability"], 2 / 3)
        self.assertEqual(
            result.iloc[0]["risk_source"],
            "lead_time_band + market_segment + distribution_channel + customer_type",
        )

    def test_sparse_specific_bucket_falls_back_to_broader_bucket(self):
        as_of = pd.Timestamp("2024-01-31")
        history = pd.DataFrame(
            [
                booking(1, "2024-01-10", "2023-12-20", is_canceled=1, cancellation_date="2024-01-01", market_segment="Direct"),
                booking(2, "2024-01-11", "2023-12-20", is_canceled=0, market_segment="Corporate"),
                booking(3, "2024-01-12", "2023-12-20", is_canceled=0, market_segment="Online TA"),
            ]
        )
        active = pd.DataFrame([booking(4, "2024-02-10", "2024-01-20", market_segment="Direct")])

        result = estimate_cancellation_probabilities(
            active,
            history,
            as_of,
            min_support=3,
            smoothing=0,
        )

        self.assertAlmostEqual(result.iloc[0]["cancellation_probability"], 1 / 3)
        self.assertEqual(result.iloc[0]["risk_source"], "lead_time_band")

    def test_snapshot_excludes_already_cancelled_and_bounds_adjusted_otb(self):
        as_of = pd.Timestamp("2024-01-31")
        bookings = pd.DataFrame(
            [
                booking(1, "2024-01-10", "2023-12-20", is_canceled=1, cancellation_date="2024-01-01"),
                booking(2, "2024-01-11", "2023-12-20", is_canceled=0),
                booking(3, "2024-02-01", "2024-01-20", is_canceled=0),
                booking(4, "2024-02-01", "2024-01-20", is_canceled=1, cancellation_date="2024-01-25"),
            ]
        )

        snapshot = calculate_otb_snapshot(bookings, as_of_date=as_of, horizon_days=1, capacity=10)
        row = snapshot.iloc[0]

        self.assertEqual(row["Live_OTB"], 1)
        self.assertAlmostEqual(row["Adjusted_OTB"], 0.5)
        self.assertAlmostEqual(row["Expected_Cancellations"], 0.5)
        self.assertGreaterEqual(row["Adjusted_OTB"], 0)
        self.assertLessEqual(row["Adjusted_OTB"], row["Live_OTB"])

    def test_fragile_otb_no_longer_overstates_pricing_occupancy(self):
        result = optimizer_node(
            {
                "target_date": "2024-02-10",
                "forecasted_occupancy": 0.70,
                "current_occupancy": 0.60,
                "raw_otb_occupancy": 0.90,
                "adjusted_otb_occupancy": 0.60,
                "expected_cancellations": 30.0,
                "competitor_price": 140.0,
                "booking_velocity": 1.0,
                "manual_demand_shock": 0.0,
                "demand_shock": 0.0,
                "local_intel_applied_shock": 0.0,
                "manual_event_text": "",
                "local_intel_estimate": {},
            }
        )

        self.assertEqual(result["pricing_breakdown"]["pricing_occupancy"], 0.70)
        self.assertEqual(result["pricing_breakdown"]["raw_otb_occupancy"], 0.90)
        self.assertEqual(result["pricing_breakdown"]["adjusted_otb_occupancy"], 0.60)

    def test_high_retention_otb_still_lifts_pricing_occupancy(self):
        result = optimizer_node(
            {
                "target_date": "2024-02-10",
                "forecasted_occupancy": 0.70,
                "current_occupancy": 0.85,
                "raw_otb_occupancy": 0.88,
                "adjusted_otb_occupancy": 0.85,
                "expected_cancellations": 3.0,
                "competitor_price": 140.0,
                "booking_velocity": 1.0,
                "manual_demand_shock": 0.0,
                "demand_shock": 0.0,
                "local_intel_applied_shock": 0.0,
                "manual_event_text": "",
                "local_intel_estimate": {},
            }
        )

        self.assertEqual(result["pricing_breakdown"]["pricing_occupancy"], 0.85)

    def test_near_zero_cancellation_risk_keeps_optimizer_price_stable(self):
        baseline_price, _ = calculate_recommended_price(
            occupancy=0.82,
            day_name="Saturday",
            competitor_price=140.0,
            booking_velocity=1.0,
        )
        adjusted_price, _, breakdown = calculate_recommended_price(
            occupancy=0.82,
            day_name="Saturday",
            competitor_price=140.0,
            booking_velocity=1.0,
            raw_otb_occupancy=0.82,
            adjusted_otb_occupancy=0.82,
            expected_cancellations=0.0,
            return_breakdown=True,
        )

        self.assertEqual(adjusted_price, baseline_price)
        self.assertEqual(breakdown["pricing_occupancy"], 0.82)


if __name__ == "__main__":
    unittest.main()
