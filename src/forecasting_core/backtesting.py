from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

import forecasting_core.legacy as legacy


class ForecastBacktester:
    """Rolling-origin backtesting and audit calculations."""

    def generate_weekly_folds(
        self,
        df: pd.DataFrame,
        horizon: int,
        min_train_days: int = legacy.DEFAULT_MIN_TRAIN_DAYS,
        step_days: int = legacy.DEFAULT_BACKTEST_STEP_DAYS,
        audit_folds: int = legacy.DEFAULT_AUDIT_FOLDS,
    ) -> pd.DataFrame:
        return legacy._generate_weekly_folds(df, horizon, min_train_days, step_days, audit_folds)

    def scenario_folds(self, df: pd.DataFrame, scenario_lags: Iterable[int], horizon: int) -> pd.DataFrame:
        return legacy._scenario_folds(df, scenario_lags, horizon)

    def metrics_from_predictions(self, predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        return legacy._metrics_from_predictions(predictions, group_cols)

    def aggregate_fold_metrics(self, fold_metrics: pd.DataFrame, split: str) -> pd.DataFrame:
        return legacy._aggregate_fold_metrics(fold_metrics, split)

    def calibrate_interval_quantiles(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        feature_profile: str = legacy.BASELINE_PROFILE,
        interval_level: float = legacy.DEFAULT_INTERVAL_LEVEL,
    ) -> tuple[pd.DataFrame, dict]:
        return legacy._calibrate_interval_quantiles(predictions, model_name, feature_profile, interval_level)

    def interval_bounds_for_lag(self, prediction: float, lag: int, interval_quantiles: dict) -> tuple[float, float]:
        return legacy._interval_bounds_for_lag(prediction, lag, interval_quantiles)

    def interval_coverage(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        feature_profile: str,
        interval_quantiles: dict,
    ) -> pd.DataFrame:
        return legacy._interval_coverage(predictions, model_name, feature_profile, interval_quantiles)

    def audit_status(
        self,
        selection_mean_wape: float,
        audit_mean_wape: float,
        drift_threshold: float = legacy.DEFAULT_AUDIT_DRIFT_THRESHOLD,
    ) -> tuple[float, str]:
        return legacy._audit_status(selection_mean_wape, audit_mean_wape, drift_threshold)
