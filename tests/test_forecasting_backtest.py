import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import (
    _aggregate_fold_metrics,
    _audit_status,
    _build_chain_training,
    _build_recursive_training,
    _calibrate_interval_quantiles,
    _feature_vector,
    _select_chain_schema,
    _select_champion_with_acceptance,
    _select_production_feature_schemas,
    _split_feature_schema,
    _generate_weekly_folds,
    _interval_bounds_for_lag,
    _unavailable_model_reason,
    ForecastChampion,
    forecast_demand,
    load_champion,
    run_backtest_detailed,
    save_champion,
)
import forecasting
import forecasting_core.legacy as forecasting_impl
from forecasting_core.algorithms import ForecastPrediction, algorithm_for_model
from forecasting_core.boruta_selector import BorutaFeatureSelector
from forecasting_core.config import ForecastRunConfig
from forecasting_core.engine import ForecastEngine
from forecasting_core.model_registry import ForecastModelRegistry


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
    def test_modular_forecast_engine_preserves_forecast_output_contract(self):
        daily = make_daily_frame(periods=430)
        engine = ForecastEngine(config=ForecastRunConfig(horizon=7))

        forecast, schema = engine.forecast_demand(
            daily,
            selected_model="seasonal_naive_7",
            horizon_days=7,
        )

        self.assertEqual(
            list(forecast.columns),
            [
                "Date",
                "Forecasted_Occupancy",
                "Min_Occupancy",
                "Max_Occupancy",
                "Competitor_Rate",
                "Selected_Model",
                "Feature_Profile",
            ],
        )
        self.assertEqual(len(forecast), 7)
        self.assertEqual(schema, [])
        self.assertEqual(set(forecast["Selected_Model"]), {"seasonal_naive_7"})

    def test_model_registry_exposes_availability_without_relabeling_sarimax(self):
        registry = ForecastModelRegistry()
        original_sarimax = forecasting_impl.SARIMAX
        try:
            forecasting_impl.SARIMAX = None

            self.assertEqual(
                registry.unavailable_model_reason("sarimax"),
                "statsmodels SARIMAX is unavailable",
            )
            self.assertNotIn("sarimax", registry.available_models(["seasonal_naive_7", "sarimax"]))
        finally:
            forecasting_impl.SARIMAX = original_sarimax

    def test_algorithm_interface_returns_prediction_object(self):
        daily = make_daily_frame(periods=100)
        history = forecasting._actuals(daily)

        prediction = algorithm_for_model("seasonal_naive_7").predict(history, horizon=7)

        self.assertIsInstance(prediction, ForecastPrediction)
        self.assertEqual(len(prediction.values), 7)
        self.assertEqual(prediction.feature_schema, [])

    def test_boruta_selector_class_keeps_chain_anchor_stability_rule(self):
        selector = BorutaFeatureSelector()
        x_train = pd.DataFrame(
            {
                "stable_feature": [0.1, 0.2, 0.3],
                "single_anchor_feature": [1.0, 1.1, 1.2],
                "h1_local_event": [0, 0, 1],
            }
        )
        y_train = np.tile(np.linspace(0.1, 0.9, 30), (3, 1))
        original_run_boruta = forecasting_impl._run_boruta
        try:
            def fake_run_boruta(x_train, y_train, anchor):
                support = {
                    "h1": [True, True],
                    "h14": [True, False],
                    "h30": [False, False],
                }[anchor]
                return pd.DataFrame(
                    {
                        "Feature": ["stable_feature", "single_anchor_feature"],
                        "Anchor": [anchor, anchor],
                        "Support": support,
                        "Support_Weak": [False, False],
                        "Rank": [1, 2],
                    }
                )

            forecasting_impl._run_boruta = fake_run_boruta
            schema, report, metadata = selector.select_chain_schema(x_train, y_train, anchors=[1, 14, 30])
        finally:
            forecasting_impl._run_boruta = original_run_boruta

        self.assertIn("stable_feature", schema)
        self.assertNotIn("single_anchor_feature", schema)
        self.assertEqual(metadata["min_anchor_count"], 2)
        self.assertTrue(report.loc[report["Feature"].eq("stable_feature"), "Selected"].all())

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

    def test_enhanced_feature_vector_adds_richer_history_without_price_signals(self):
        daily = make_daily_frame(periods=430)
        daily["Booking_Pace"] = np.linspace(0.1, 0.5, len(daily))
        daily["Bookings_Created"] = np.arange(len(daily))
        daily["Cancellations"] = np.arange(len(daily)) % 7
        future = pd.DataFrame({"Date": [daily["Date"].iloc[-1] + pd.Timedelta(days=1)], "Local_Event": [1]})

        features = _feature_vector(daily, future, horizon=1, feature_profile="boruta_selected")

        for expected in [
            "lag_364",
            "roll_min_28",
            "roll_max_28",
            "roll_slope_28",
            "trend_projection_30_from_56",
            "wow_mean_diff_7",
            "yoy_level_diff_364",
            "dow_seasonal_index",
            "booking_pace_mean_28",
            "bookings_created_sum_28",
            "cancellations_std_14",
        ]:
            self.assertIn(expected, features)
        self.assertNotIn("ADR", features)
        self.assertNotIn("RevPAR", features)
        self.assertEqual(features["h1_local_event"], 1.0)

    def test_enhanced_feature_vector_uses_only_history_for_rolling_features(self):
        daily = make_daily_frame(periods=430)
        baseline_future = pd.DataFrame({"Date": [daily["Date"].iloc[-1] + pd.Timedelta(days=1)], "Local_Event": [0]})
        altered_future = pd.DataFrame({"Date": [daily["Date"].iloc[-1] + pd.Timedelta(days=1)], "Local_Event": [1]})

        baseline = _feature_vector(daily, baseline_future, horizon=1, feature_profile="boruta_selected")
        altered = _feature_vector(daily, altered_future, horizon=1, feature_profile="boruta_selected")

        self.assertEqual(baseline["roll_mean_28"], altered["roll_mean_28"])
        self.assertEqual(baseline["trend_projection_30_from_56"], altered["trend_projection_30_from_56"])
        self.assertNotEqual(baseline["h1_local_event"], altered["h1_local_event"])

    def test_chain_selector_keeps_features_selected_in_two_of_three_anchors(self):
        x_train = pd.DataFrame(
            {
                "stable_feature": [0.1, 0.2, 0.3],
                "single_anchor_feature": [1.0, 1.1, 1.2],
                "h1_local_event": [0, 0, 1],
            }
        )
        y_train = np.tile(np.linspace(0.1, 0.9, 30), (3, 1))

        original_run_boruta = forecasting._run_boruta
        try:
            def fake_run_boruta(x_train, y_train, anchor):
                support = {
                    "h1": [True, True],
                    "h14": [True, False],
                    "h30": [False, False],
                }[anchor]
                return pd.DataFrame(
                    {
                        "Feature": ["stable_feature", "single_anchor_feature"],
                        "Anchor": [anchor, anchor],
                        "Support": support,
                        "Support_Weak": [False, False],
                        "Rank": [1, 2],
                    }
                )

            forecasting._run_boruta = fake_run_boruta
            schema, report, metadata = _select_chain_schema(x_train, y_train, anchors=[1, 14, 30])
        finally:
            forecasting._run_boruta = original_run_boruta

        self.assertIn("stable_feature", schema)
        self.assertNotIn("single_anchor_feature", schema)
        self.assertIn("h1_local_event", schema)
        self.assertEqual(metadata["min_anchor_count"], 2)
        self.assertTrue(report.loc[report["Feature"].eq("stable_feature"), "Selected"].all())

    def test_forced_keep_features_survive_even_when_boruta_rejects_them(self):
        x_train = pd.DataFrame(
            {
                "lag_1": [0.1, 0.2, 0.3, 0.4],
                "lag_7": [0.2, 0.3, 0.4, 0.5],
                "new_signal": [1.0, 1.1, 1.2, 1.3],
                "h1_local_event": [0, 0, 1, 0],
            }
        )
        y_train = np.array([0.2, 0.3, 0.4, 0.5])
        original_run_boruta = forecasting._run_boruta
        try:
            def fake_run_boruta(x_train, y_train, anchor):
                return pd.DataFrame(
                    {
                        "Feature": ["new_signal"],
                        "Anchor": [anchor],
                        "Support": [False],
                        "Support_Weak": [False],
                        "Rank": [5],
                        "Force_Kept": [False],
                    }
                )

            forecasting._run_boruta = fake_run_boruta
            schema, report, metadata = forecasting._select_recursive_schema(x_train, y_train)
        finally:
            forecasting._run_boruta = original_run_boruta

        self.assertIn("lag_1", schema)
        self.assertIn("lag_7", schema)
        self.assertNotIn("new_signal", schema)
        self.assertIn("h1_local_event", schema)
        self.assertEqual(metadata["force_kept_features"], ["lag_1", "lag_7"])
        forced_rows = report[report["Force_Kept"].fillna(False)]
        self.assertEqual(set(forced_rows["Feature"]), {"lag_1", "lag_7"})
        self.assertTrue(forced_rows["Selected"].all())

    def test_production_schema_selection_runs_once_per_strategy_not_per_fold(self):
        daily = make_daily_frame(periods=430)
        calls = {"recursive": 0, "chain": 0}
        original_recursive = forecasting._select_recursive_schema
        original_chain = forecasting._select_chain_schema
        original_predict = forecasting.predict_model
        try:
            def fake_recursive(x_train, y_train):
                calls["recursive"] += 1
                return ["lag_1", "h1_local_event"], pd.DataFrame(
                    {
                        "Feature": ["lag_1"],
                        "Anchor": ["recursive"],
                        "Support": [True],
                        "Support_Weak": [False],
                        "Rank": [1],
                        "Strategy": ["recursive_ml"],
                        "Selection_Frequency": [1],
                        "Selected": [True],
                        "Selection_Status": ["strong_support"],
                    }
                ), {}

            def fake_chain(x_train, y_train, anchors=forecasting.CHAIN_BORUTA_ANCHORS):
                calls["chain"] += 1
                return ["lag_1", "h1_local_event"], pd.DataFrame(
                    {
                        "Feature": ["lag_1"],
                        "Anchor": ["h1"],
                        "Support": [True],
                        "Support_Weak": [False],
                        "Rank": [1],
                        "Strategy": ["regressor_chain"],
                        "Selection_Frequency": [2],
                        "Selected": [True],
                        "Selection_Status": ["strong_support"],
                    }
                ), {}

            forecasting._select_recursive_schema = fake_recursive
            forecasting._select_chain_schema = fake_chain
            forecasting.predict_model = lambda history, horizon, model_name, feature_profile="statistical", selected_schema=None: (
                np.repeat(0.5, horizon),
                selected_schema or ["lag_1", "h1_local_event"],
            )
            result = run_backtest_detailed(
                daily,
                models=["random_forest_recursive", "random_forest_chain"],
                horizon=7,
                min_train_days=365,
                step_days=7,
                audit_folds=1,
                return_feature_artifacts=True,
            )
        finally:
            forecasting._select_recursive_schema = original_recursive
            forecasting._select_chain_schema = original_chain
            forecasting.predict_model = original_predict

        predictions = result[3]
        self.assertEqual(calls, {"recursive": 1, "chain": 1})
        self.assertEqual(set(predictions["Feature_Profile"]), {"boruta_selected"})

    def test_split_feature_schema_distinguishes_historical_and_mandatory_features(self):
        historical, mandatory = _split_feature_schema(["lag_1", "roll_mean_7", "h1_local_event", "h1_dow_sin"])

        self.assertEqual(historical, ["lag_1", "roll_mean_7"])
        self.assertEqual(mandatory, ["h1_local_event", "h1_dow_sin"])

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

    def test_sarimax_unavailability_is_explicit_when_dependency_is_missing(self):
        original_sarimax = forecasting.SARIMAX
        try:
            forecasting.SARIMAX = None
            self.assertEqual(
                _unavailable_model_reason("sarimax"),
                "statsmodels SARIMAX is unavailable",
            )
        finally:
            forecasting.SARIMAX = original_sarimax

    def test_backtest_skips_unavailable_sarimax_instead_of_relabeling_fallback(self):
        original_sarimax = forecasting.SARIMAX
        try:
            forecasting.SARIMAX = None
            daily = make_daily_frame(periods=430)
            overall, _, _, predictions, _, _ = run_backtest_detailed(
                daily,
                models=["seasonal_naive_7", "sarimax"],
                horizon=7,
                min_train_days=365,
                step_days=7,
                audit_folds=1,
            )

            self.assertEqual(set(predictions["Model"]), {"seasonal_naive_7"})
            self.assertEqual(set(overall["Model"]), {"seasonal_naive_7"})
        finally:
            forecasting.SARIMAX = original_sarimax

    def test_champion_serialization_preserves_feature_profile_metadata(self):
        champion = ForecastChampion(
            model="random_forest_recursive",
            strategy="recursive_ml",
            horizon=30,
            selected_at="2026-05-18T00:00:00",
            metrics={"WAPE": 10.0},
            feature_schema=["lag_1", "h1_local_event"],
            feature_profile="boruta_selected",
            selected_historical_features=["lag_1"],
            mandatory_features=["h1_local_event"],
            feature_selection_metadata={"selector": "boruta"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "champion.json"
            save_champion(champion, str(path))
            loaded = load_champion(str(path))

        self.assertEqual(loaded.feature_profile, "boruta_selected")
        self.assertEqual(loaded.selected_historical_features, ["lag_1"])
        self.assertEqual(loaded.mandatory_features, ["h1_local_event"])
        self.assertEqual(loaded.feature_selection_metadata, {"selector": "boruta"})

    def test_forecast_uses_saved_schema_without_reselecting_features(self):
        daily = make_daily_frame(periods=430)
        original_recursive = forecasting._select_recursive_schema
        original_predict = forecasting.predict_model
        try:
            def fail_if_called(*args, **kwargs):
                raise AssertionError("forecast should use saved champion schema")

            forecasting._select_recursive_schema = fail_if_called
            forecasting.predict_model = lambda history, horizon, model_name, feature_profile="statistical", selected_schema=None: (
                np.repeat(0.5, horizon),
                selected_schema or [],
            )
            forecast, schema = forecast_demand(
                daily,
                selected_model="random_forest_recursive",
                horizon_days=7,
                feature_profile="boruta_selected",
                feature_schema=["lag_1", "h1_local_event"],
            )
        finally:
            forecasting._select_recursive_schema = original_recursive
            forecasting.predict_model = original_predict

        self.assertEqual(schema, ["lag_1", "h1_local_event"])
        self.assertEqual(set(forecast["Feature_Profile"]), {"boruta_selected"})

    def test_single_boruta_profile_has_no_baseline_comparator(self):
        overall = pd.DataFrame(
            [
                {
                    "Feature_Profile": "boruta_selected",
                    "Model": "random_forest_recursive",
                    "Strategy": "recursive_ml",
                    "WAPE": 9.0,
                    "Abs_Bias": 0.1,
                    "RMSE": 0.1,
                    "Complexity": 4,
                },
            ]
        )
        audit = pd.DataFrame(
            [
                {"Feature_Profile": "boruta_selected", "Model": "random_forest_recursive", "WAPE": 12.0},
            ]
        )

        champion, metadata = _select_champion_with_acceptance(overall, audit, horizon=30)

        self.assertEqual(champion.feature_profile, "boruta_selected")
        self.assertTrue(metadata["accepted"])
        self.assertEqual(metadata["reason"], "single_profile_no_baseline_comparator")


if __name__ == "__main__":
    unittest.main()
