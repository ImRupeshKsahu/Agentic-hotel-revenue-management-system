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
    stay_nights=1,
    market_segment="Direct",
    distribution_channel="Direct",
    customer_type="Transient",
):
    arrival_date = pd.Timestamp(arrival_date)
    return {
        "booking_id": booking_id,
        "arrival_date": arrival_date,
        "departure_date": arrival_date + pd.Timedelta(days=stay_nights),
        "booking_date": pd.Timestamp(booking_date),
        "stay_nights": stay_nights,
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
            "remaining_days_band + lead_time_band + market_segment + distribution_channel + customer_type",
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
        self.assertEqual(result.iloc[0]["risk_source"], "remaining_days_band + lead_time_band")

    def test_remaining_risk_falls_as_arrival_gets_closer(self):
        as_of = pd.Timestamp("2024-01-31")
        history = pd.DataFrame(
            [
                booking(1, "2024-01-31", "2023-12-01", is_canceled=1, cancellation_date="2024-01-10", lead_time=61),
                booking(2, "2024-01-31", "2023-12-01", is_canceled=1, cancellation_date="2024-01-20", lead_time=61),
                booking(3, "2024-01-31", "2023-12-01", is_canceled=1, cancellation_date="2024-01-25", lead_time=61),
                booking(4, "2024-01-31", "2023-12-01", lead_time=61),
            ]
        )
        active = pd.DataFrame(
            [
                booking(5, "2024-02-01", "2023-12-01", lead_time=61),
                booking(6, "2024-02-20", "2023-12-01", lead_time=61),
            ]
        )

        result = estimate_cancellation_probabilities(active, history, as_of, min_support=1, smoothing=0)

        one_day_risk = result.iloc[0]["cancellation_probability"]
        fifteen_to_thirty_day_risk = result.iloc[1]["cancellation_probability"]
        self.assertLess(one_day_risk, fifteen_to_thirty_day_risk)

    def test_arrived_stayovers_receive_zero_cancellation_probability(self):
        as_of = pd.Timestamp("2024-01-31")
        history = pd.DataFrame(
            [
                booking(1, "2024-01-10", "2023-12-20", is_canceled=1, cancellation_date="2024-01-01"),
                booking(2, "2024-01-11", "2023-12-20"),
            ]
        )
        active = pd.DataFrame([booking(3, "2024-01-30", "2024-01-01", stay_nights=3)])

        result = estimate_cancellation_probabilities(active, history, as_of)

        self.assertEqual(result.iloc[0]["remaining_days_band"], "arrived")
        self.assertEqual(result.iloc[0]["cancellation_probability"], 0.0)
        self.assertEqual(result.iloc[0]["risk_source"], "arrived")

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
        self.assertAlmostEqual(row["Adjusted_OTB"], 1.0)
        self.assertAlmostEqual(row["Expected_Cancellations"], 0.0)
        self.assertGreaterEqual(row["Adjusted_OTB"], 0)
        self.assertLessEqual(row["Adjusted_OTB"], row["Live_OTB"])

    def test_snapshot_splits_stayovers_from_future_arrivals(self):
        as_of = pd.Timestamp("2024-01-31")
        bookings = pd.DataFrame(
            [
                booking(1, "2024-01-30", "2024-01-01", stay_nights=3),
                booking(2, "2024-02-01", "2024-01-20"),
                booking(3, "2024-01-10", "2023-12-20", is_canceled=1, cancellation_date="2024-01-01"),
                booking(4, "2024-01-11", "2023-12-20"),
            ]
        )

        snapshot = calculate_otb_snapshot(bookings, as_of_date=as_of, horizon_days=1, capacity=10)
        row = snapshot.iloc[0]

        self.assertEqual(row["Stayover_OTB"], 1)
        self.assertEqual(row["Future_Arrival_OTB"], 1)
        self.assertAlmostEqual(row["Adjusted_OTB"], row["Stayover_OTB"] + (row["Future_Arrival_OTB"] - row["Expected_Cancellations"]))
        self.assertLessEqual(row["Expected_Cancellations"], row["Future_Arrival_OTB"])

    def test_gross_pace_uses_uncapped_otb_while_display_otb_stays_capped(self):
        as_of = pd.Timestamp("2024-01-31")
        current_stay = "2024-02-01"
        historical_stay = "2023-02-01"
        rows = []
        for booking_id in range(1, 13):
            rows.append(booking(booking_id, current_stay, "2024-01-20"))
        for booking_id in range(13, 23):
            rows.append(booking(booking_id, historical_stay, "2023-01-20"))

        snapshot = calculate_otb_snapshot(pd.DataFrame(rows), as_of_date=as_of, horizon_days=1, capacity=10)
        row = snapshot.iloc[0]

        self.assertEqual(row["Live_OTB"], 10)
        self.assertEqual(row["Gross_OTB"], 12)
        self.assertEqual(row["Historical_Avg_OTB"], 10)
        self.assertEqual(row["Gross_Pace_Index"], 1.10)
        self.assertEqual(row["Booking_Velocity"], row["Gross_Pace_Index"])

    def test_snapshot_emits_richer_pace_family(self):
        as_of = pd.Timestamp("2024-01-31")
        bookings = pd.DataFrame(
            [
                booking(1, "2024-02-01", "2024-01-20"),
                booking(2, "2024-02-01", "2024-01-28"),
                booking(3, "2023-02-01", "2023-01-20"),
            ]
        )

        snapshot = calculate_otb_snapshot(bookings, as_of_date=as_of, horizon_days=1, capacity=10)
        row = snapshot.iloc[0]

        for column in [
            "Gross_Pace_Index",
            "Retained_Pace_Index",
            "Pickup_Trend_Index",
            "Pricing_Pace_Index",
            "Pace_Confidence",
        ]:
            self.assertIn(column, snapshot.columns)
        self.assertEqual(row["Pace_Confidence"], "high")

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

    def test_sold_out_raw_otb_applies_competitor_parity_floor(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=0.932,
            day_name="Friday",
            competitor_price=138.41,
            booking_velocity=1.0,
            raw_otb_occupancy=1.0,
            adjusted_otb_occupancy=0.6504,
            expected_cancellations=82.86,
            return_breakdown=True,
        )

        self.assertGreaterEqual(price, 138.41)
        self.assertTrue(breakdown["sold_out"])
        self.assertEqual(breakdown["pricing_regime"], "sold_out_protect_rate")
        self.assertTrue(breakdown["sold_out_floor_applied"])
        self.assertTrue(breakdown["material_retention_gap"])
        self.assertTrue(
            any("retained occupancy is materially lower" in flag for flag in breakdown["review_flags"])
        )

    def test_sold_out_floor_does_not_reduce_stronger_optimizer_price(self):
        price, _, breakdown = calculate_recommended_price(
            occupancy=1.0,
            day_name="Friday",
            competitor_price=80.0,
            booking_velocity=1.0,
            raw_otb_occupancy=1.0,
            adjusted_otb_occupancy=0.98,
            expected_cancellations=2.0,
            return_breakdown=True,
        )

        self.assertGreaterEqual(price, breakdown["selected_candidate"]["price"])
        self.assertFalse(breakdown["sold_out_floor_applied"])


if __name__ == "__main__":
    unittest.main()
