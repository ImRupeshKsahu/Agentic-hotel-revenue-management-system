import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import (
    _aggregate_fold_metrics,
    _audit_status,
    _build_recursive_training,
    _calibrate_interval_quantiles,
    _feature_vector,
    _generate_weekly_folds,
    _interval_bounds_for_lag,
)


def make_daily_frame(start="2015-07-01", periods=793):
    dates = pd.date_range(start, periods=periods, freq="D")
    occupancy = 0.55 + 0.12 * np.sin(np.arange(periods) * 2 * np.pi / 7)
    return pd.DataFrame(
        {
            "Date": dates,
            "Occupancy_Rate": occupancy,
            "Competitor_Rate": 120.0,
            "Booking_Pace": 0.1,
            "Cancellations": 1,
            "Bookings_Created": 2,
            "Local_Event": 0,
        }
    )


class ForecastingBacktestTests(unittest.TestCase):
    def test_weekly_folds_are_full_horizon_and_split_49_8(self):
        daily = make_daily_frame()
        folds = _generate_weekly_folds(daily, horizon=30, min_train_days=365, step_days=7, audit_folds=8)

        self.assertEqual(len(folds), 57)
        self.assertEqual(int(folds["Split"].eq("selection").sum()), 49)
        self.assertEqual(int(folds["Split"].eq("audit").sum()), 8)
        self.assertEqual(folds["Cutoff"].iloc[0], pd.Timestamp("2016-07-05"))
        self.assertEqual(folds["Cutoff"].iloc[-1], pd.Timestamp("2017-08-01"))
        self.assertTrue(((folds["Validation_End"] - folds["Validation_Start"]).dt.days + 1).eq(30).all())
        self.assertTrue((folds["Cutoff"] - folds["Train_Start"]).dt.days.ge(365).all())

    def test_recursive_features_exclude_synthetic_competitor_rate(self):
        daily = make_daily_frame(periods=100)
        daily.loc[91:, "Competitor_Rate"] = 999.0

        x_train, _, _ = _build_recursive_training(daily, min_history=90)

        self.assertNotIn("recent_competitor_rate", x_train.columns)
        self.assertNotIn("h1_competitor_rate", x_train.columns)

    def test_feature_vector_keeps_future_known_event_but_not_competitor_rate(self):
        daily = make_daily_frame(periods=100)
        future = pd.DataFrame(
            {
                "Date": [daily["Date"].iloc[-1] + pd.Timedelta(days=1)],
                "Local_Event": [1],
                "Competitor_Rate": [999.0],
            }
        )

        features = _feature_vector(daily, future, horizon=1)

        self.assertEqual(features["h1_local_event"], 1.0)
        self.assertNotIn("recent_competitor_rate", features)
        self.assertNotIn("h1_competitor_rate", features)

    def test_mean_fold_wape_averages_folds_equally(self):
        fold_metrics = pd.DataFrame(
            [
                {"Split": "selection", "Fold_ID": "fold_001", "Model": "a", "Strategy": "s", "Observations": 30, "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1, "WAPE": 2, "Bias": 0, "Accuracy": 98, "Volatility": 0, "Stability": 1},
                {"Split": "selection", "Fold_ID": "fold_002", "Model": "a", "Strategy": "s", "Observations": 10, "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1, "WAPE": 8, "Bias": 0, "Accuracy": 92, "Volatility": 0, "Stability": 1},
            ]
        )

        overall = _aggregate_fold_metrics(fold_metrics, split="selection")

        self.assertEqual(float(overall.iloc[0]["WAPE"]), 5.0)
        self.assertEqual(int(overall.iloc[0]["Folds"]), 2)

    def test_audit_status_uses_relative_drift_threshold(self):
        ok_ratio, ok_status = _audit_status(4.0, 5.0, drift_threshold=0.25)
        bad_ratio, bad_status = _audit_status(4.0, 5.1, drift_threshold=0.25)

        self.assertEqual(ok_ratio, 1.25)
        self.assertEqual(ok_status, "ok")
        self.assertEqual(round(bad_ratio, 3), 1.275)
        self.assertEqual(bad_status, "recent_degradation_flagged")

    def test_interval_quantiles_are_lag_specific(self):
        predictions = pd.DataFrame(
            {
                "Model": ["m"] * 6,
                "Lag": [1, 1, 1, 2, 2, 2],
                "Actual": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "Predicted": [0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
            }
        )
        _, quantiles = _calibrate_interval_quantiles(predictions, "m", interval_level=0.90)

        lag1_bounds = _interval_bounds_for_lag(0.5, 1, quantiles)
        lag2_bounds = _interval_bounds_for_lag(0.6, 2, quantiles)

        self.assertNotEqual(lag1_bounds, lag2_bounds)
        self.assertTrue(0 <= lag1_bounds[0] <= lag1_bounds[1] <= 1)
        self.assertTrue(0 <= lag2_bounds[0] <= lag2_bounds[1] <= 1)


if __name__ == "__main__":
    unittest.main()
