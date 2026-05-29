"""Compatibility facade for the modular forecasting package.

New code should prefer the classes under ``forecasting_core``. This module keeps
the historical function and constant names stable for scripts, tests, and older
Streamlit code.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import pandas as pd

from forecasting_core.algorithms import (
    ChainMLAlgorithm,
    ForecastAlgorithm,
    ForecastPrediction,
    RecursiveMLAlgorithm,
    StatisticalAlgorithm,
    algorithm_for_model,
)
from forecasting_core.artifacts import ArtifactStore, ForecastChampion
from forecasting_core.backtesting import ForecastBacktester
from forecasting_core.boruta_selector import BorutaFeatureSelector
from forecasting_core.config import (
    FeatureSelectionConfig,
    ForecastRunConfig,
    ModelCompetitionConfig,
)
from forecasting_core.engine import ForecastEngine
from forecasting_core.hyperparameter_tuning import ForecastHyperparameterTuner, HyperparameterTuningConfig
import forecasting_core.legacy as _legacy
from forecasting_core.model_registry import ForecastModelRegistry
from forecasting_core.plots import ForecastPlotter


DEFAULT_HORIZON = _legacy.DEFAULT_HORIZON
DEFAULT_SCENARIO_LAGS = _legacy.DEFAULT_SCENARIO_LAGS
DEFAULT_BACKTEST_STEP_DAYS = _legacy.DEFAULT_BACKTEST_STEP_DAYS
DEFAULT_MIN_TRAIN_DAYS = _legacy.DEFAULT_MIN_TRAIN_DAYS
DEFAULT_AUDIT_FOLDS = _legacy.DEFAULT_AUDIT_FOLDS
DEFAULT_INTERVAL_LEVEL = _legacy.DEFAULT_INTERVAL_LEVEL
DEFAULT_AUDIT_DRIFT_THRESHOLD = _legacy.DEFAULT_AUDIT_DRIFT_THRESHOLD
DEFAULT_HYPERPARAM_TRIALS = _legacy.DEFAULT_HYPERPARAM_TRIALS
DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS = _legacy.DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS
DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP = _legacy.DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP
MODEL_SELECTION_OBJECTIVE = _legacy.MODEL_SELECTION_OBJECTIVE
MODEL_SELECTION_MAE_TIE_THRESHOLD_PP = _legacy.MODEL_SELECTION_MAE_TIE_THRESHOLD_PP
BASELINE_PROFILE = _legacy.BASELINE_PROFILE
ENHANCED_PROFILE = _legacy.ENHANCED_PROFILE
DEFAULT_FEATURE_PROFILES = _legacy.DEFAULT_FEATURE_PROFILES
CHAIN_BORUTA_ANCHORS = _legacy.CHAIN_BORUTA_ANCHORS
CHAIN_BORUTA_MIN_ANCHORS = _legacy.CHAIN_BORUTA_MIN_ANCHORS
BORUTA_MAX_ITER = _legacy.BORUTA_MAX_ITER
BORUTA_TREE_COUNT = _legacy.BORUTA_TREE_COUNT
BORUTA_PERC = _legacy.BORUTA_PERC

STATISTICAL_MODELS = _legacy.STATISTICAL_MODELS
DEFAULT_STATISTICAL_MODELS = _legacy.DEFAULT_STATISTICAL_MODELS
ML_RECURSIVE_MODELS = _legacy.ML_RECURSIVE_MODELS
ML_CHAIN_MODELS = _legacy.ML_CHAIN_MODELS
EXPERIMENTAL_MODELS = _legacy.EXPERIMENTAL_MODELS
SUPPORTED_MODELS = _legacy.SUPPORTED_MODELS
DEFAULT_MODELS = _legacy.DEFAULT_MODELS
MODEL_COMPLEXITY = _legacy.MODEL_COMPLEXITY
SARIMAX_CANDIDATES = _legacy.SARIMAX_CANDIDATES

ExtraTreesRegressor = _legacy.ExtraTreesRegressor
RandomForestRegressor = _legacy.RandomForestRegressor
Ridge = _legacy.Ridge
ElasticNet = _legacy.ElasticNet
RegressorChain = _legacy.RegressorChain
make_pipeline = _legacy.make_pipeline
StandardScaler = _legacy.StandardScaler
XGBRegressor = _legacy.XGBRegressor
BorutaPy = _legacy.BorutaPy
ExponentialSmoothing = _legacy.ExponentialSmoothing
SARIMAX = _legacy.SARIMAX

_FEATURE_ENGINEER = _legacy._FEATURE_ENGINEER
_DEFAULT_ENGINE = ForecastEngine()


def _sync_facade_overrides() -> None:
    """Mirror deliberate test/runtime monkeypatches onto the private implementation."""
    dependency_names = [
        "ExtraTreesRegressor",
        "RandomForestRegressor",
        "Ridge",
        "ElasticNet",
        "RegressorChain",
        "make_pipeline",
        "StandardScaler",
        "XGBRegressor",
        "BorutaPy",
        "ExponentialSmoothing",
        "SARIMAX",
    ]
    for name in dependency_names:
        setattr(_legacy, name, globals()[name])

    patchable_functions = [
        ("_run_boruta", _RUN_BORUTA_WRAPPER, _ORIGINAL_RUN_BORUTA),
        ("_select_recursive_schema", _SELECT_RECURSIVE_SCHEMA_WRAPPER, _ORIGINAL_SELECT_RECURSIVE_SCHEMA),
        ("_select_chain_schema", _SELECT_CHAIN_SCHEMA_WRAPPER, _ORIGINAL_SELECT_CHAIN_SCHEMA),
        ("predict_model", _PREDICT_MODEL_WRAPPER, _ORIGINAL_PREDICT_MODEL),
    ]
    for name, wrapper, original in patchable_functions:
        current = globals()[name]
        setattr(_legacy, name, original if current is wrapper else current)


def _unavailable_model_reason(model_name: str) -> Optional[str]:
    _sync_facade_overrides()
    return _legacy._unavailable_model_reason(model_name)


def available_models(models: Iterable[str]) -> list[str]:
    _sync_facade_overrides()
    return _legacy.available_models(models)


def _actuals(daily_df: pd.DataFrame) -> pd.DataFrame:
    return _legacy._actuals(daily_df)


def calculate_forecast_metrics(actual, predicted) -> dict:
    return _legacy.calculate_forecast_metrics(actual, predicted)


def _strategy_for_model(model_name: str) -> str:
    return _legacy._strategy_for_model(model_name)


def _future_dates(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return _legacy._future_dates(history, horizon)


def _sarimax_exog(df: pd.DataFrame) -> pd.DataFrame:
    return _legacy._sarimax_exog(df)


def _calendar_features(date: pd.Timestamp, prefix: str) -> dict:
    return _legacy._calendar_features(date, prefix)


def _series_value_or_mean(series: pd.Series, lag: int) -> float:
    return _legacy._series_value_or_mean(series, lag)


def _rolling_slope(series: pd.Series, window: int) -> float:
    return _legacy._rolling_slope(series, window)


def _seasonal_index(history: pd.DataFrame, group: str, value) -> float:
    return _legacy._seasonal_index(history, group, value)


def _mandatory_feature_names(columns: Iterable[str]) -> list[str]:
    return _legacy._mandatory_feature_names(columns)


def _split_feature_schema(feature_schema: Iterable[str]) -> tuple[list[str], list[str]]:
    return _legacy._split_feature_schema(feature_schema)


def _baseline_feature_vector(history: pd.DataFrame, future: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    return _legacy._baseline_feature_vector(history, future, horizon)


def _enhanced_feature_vector(history: pd.DataFrame, future: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    return _legacy._enhanced_feature_vector(history, future, horizon)


def _feature_vector(
    history: pd.DataFrame,
    future: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    feature_profile: str = BASELINE_PROFILE,
) -> dict:
    return _legacy._feature_vector(history, future, horizon, feature_profile)


def _build_chain_training(
    history: pd.DataFrame,
    horizon: int,
    min_history: int = 90,
    feature_profile: str = BASELINE_PROFILE,
):
    return _legacy._build_chain_training(history, horizon, min_history, feature_profile)


def _build_recursive_training(
    history: pd.DataFrame,
    min_history: int = 90,
    feature_profile: str = BASELINE_PROFILE,
):
    return _legacy._build_recursive_training(history, min_history, feature_profile)


def _boruta_available() -> bool:
    _sync_facade_overrides()
    return _legacy._boruta_available()


def _boruta_unavailable_reason() -> Optional[str]:
    _sync_facade_overrides()
    return _legacy._boruta_unavailable_reason()


def _run_boruta(x_train: pd.DataFrame, y_train, anchor: str) -> pd.DataFrame:
    _sync_facade_overrides()
    return _legacy._run_boruta(x_train, y_train, anchor)


def _force_keep_features(columns: Iterable[str]) -> list[str]:
    return _legacy._force_keep_features(columns)


def _force_keep_report(features: Iterable[str], anchor: str, strategy: str) -> pd.DataFrame:
    return _legacy._force_keep_report(features, anchor, strategy)


def _stable_features_from_boruta(report: pd.DataFrame, min_anchor_count: int) -> tuple[list[str], str]:
    return _legacy._stable_features_from_boruta(report, min_anchor_count)


def _select_recursive_schema(x_train: pd.DataFrame, y_train) -> tuple[list[str], pd.DataFrame, dict]:
    _sync_facade_overrides()
    return _legacy._select_recursive_schema(x_train, y_train)


def _select_chain_schema(
    x_train: pd.DataFrame,
    y_train,
    anchors: Iterable[int] = CHAIN_BORUTA_ANCHORS,
) -> tuple[list[str], pd.DataFrame, dict]:
    _sync_facade_overrides()
    return _legacy._select_chain_schema(x_train, y_train, anchors)


def _select_production_feature_schemas(
    history: pd.DataFrame,
    horizon: int,
    model_specs: Iterable[dict],
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    _sync_facade_overrides()
    return _legacy._select_production_feature_schemas(history, horizon, model_specs)


def _base_estimator(model_name: str, tuned_params: Optional[dict] = None):
    _sync_facade_overrides()
    return _legacy._base_estimator(model_name, tuned_params=tuned_params)


def _predict_statistical(history: pd.DataFrame, horizon: int, model_name: str):
    _sync_facade_overrides()
    return _legacy._predict_statistical(history, horizon, model_name)


def _predict_chain(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    _sync_facade_overrides()
    return _legacy._predict_chain(history, horizon, model_name, feature_profile, selected_schema, tuned_params)


def _predict_recursive(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    _sync_facade_overrides()
    return _legacy._predict_recursive(history, horizon, model_name, feature_profile, selected_schema, tuned_params)


def predict_model(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    _sync_facade_overrides()
    return _legacy.predict_model(history, horizon, model_name, feature_profile, selected_schema, tuned_params)


def _generate_weekly_folds(
    df: pd.DataFrame,
    horizon: int,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    step_days: int = DEFAULT_BACKTEST_STEP_DAYS,
    audit_folds: int = DEFAULT_AUDIT_FOLDS,
) -> pd.DataFrame:
    return _legacy._generate_weekly_folds(df, horizon, min_train_days, step_days, audit_folds)


def _scenario_folds(df: pd.DataFrame, scenario_lags: Iterable[int], horizon: int) -> pd.DataFrame:
    return _legacy._scenario_folds(df, scenario_lags, horizon)


def _metrics_from_predictions(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return _legacy._metrics_from_predictions(predictions, group_cols)


def _with_feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    return _legacy._with_feature_profile(df)


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    return _legacy._aggregate_fold_metrics(fold_metrics, split)


def _model_specs(models: Iterable[str]) -> list[dict]:
    _sync_facade_overrides()
    return _legacy._model_specs(models)


def _feature_manifest(
    history: pd.DataFrame,
    horizon: int,
    champion_profile: str,
    champion_schema: Iterable[str],
) -> pd.DataFrame:
    _sync_facade_overrides()
    return _legacy._feature_manifest(history, horizon, champion_profile, champion_schema)


def _calibrate_interval_quantiles(
    predictions: pd.DataFrame,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> tuple[pd.DataFrame, dict]:
    return _legacy._calibrate_interval_quantiles(predictions, model_name, feature_profile, interval_level)


def _interval_bounds_for_lag(prediction: float, lag: int, interval_quantiles: dict) -> tuple[float, float]:
    return _legacy._interval_bounds_for_lag(prediction, lag, interval_quantiles)


def _interval_coverage(
    predictions: pd.DataFrame,
    model_name: str,
    feature_profile: str,
    interval_quantiles: dict,
) -> pd.DataFrame:
    return _legacy._interval_coverage(predictions, model_name, feature_profile, interval_quantiles)


def _audit_status(
    selection_mean_wape: float,
    audit_mean_wape: float,
    drift_threshold: float = DEFAULT_AUDIT_DRIFT_THRESHOLD,
) -> tuple[float, str]:
    return _legacy._audit_status(selection_mean_wape, audit_mean_wape, drift_threshold)


def run_backtest_detailed(
    daily_df: pd.DataFrame,
    models: Optional[Iterable[str]] = None,
    horizon: int = DEFAULT_HORIZON,
    scenario_lags: Optional[Iterable[int]] = None,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    step_days: int = DEFAULT_BACKTEST_STEP_DAYS,
    audit_folds: int = DEFAULT_AUDIT_FOLDS,
    return_feature_artifacts: bool = False,
):
    _sync_facade_overrides()
    return _legacy.run_backtest_detailed(
        daily_df,
        models=models,
        horizon=horizon,
        scenario_lags=scenario_lags,
        min_train_days=min_train_days,
        step_days=step_days,
        audit_folds=audit_folds,
        return_feature_artifacts=return_feature_artifacts,
    )


def select_champion(overall_metrics: pd.DataFrame, horizon: int, feature_schema: list[str] | None = None) -> ForecastChampion:
    return _legacy.select_champion(overall_metrics, horizon, feature_schema)


def _select_champion_with_acceptance(
    overall_metrics: pd.DataFrame,
    audit_summary: pd.DataFrame,
    horizon: int,
) -> tuple[ForecastChampion, dict]:
    return _legacy._select_champion_with_acceptance(overall_metrics, audit_summary, horizon)


def save_champion(champion: ForecastChampion, path: str):
    return _legacy.save_champion(champion, path)


def _safe_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    return _legacy._safe_to_csv(df, path, index)


def load_champion(path: str, default_horizon: int = DEFAULT_HORIZON) -> ForecastChampion:
    return _legacy.load_champion(path, default_horizon)


def forecast_demand(
    daily_df: pd.DataFrame,
    selected_model: str = "seasonal_naive_7",
    horizon_days: int = DEFAULT_HORIZON,
    interval_quantiles: Optional[dict] = None,
    feature_profile: str = BASELINE_PROFILE,
    feature_schema: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    _sync_facade_overrides()
    return _legacy.forecast_demand(
        daily_df,
        selected_model=selected_model,
        horizon_days=horizon_days,
        interval_quantiles=interval_quantiles,
        feature_profile=feature_profile,
        feature_schema=feature_schema,
    )


def _plot_best_model_forecast(history: pd.DataFrame, forecast: pd.DataFrame, best_model: str, output_path: str):
    return _legacy._plot_best_model_forecast(history, forecast, best_model, output_path)


def _plot_lag_metrics(lag_metrics: pd.DataFrame, plots_dir: str):
    return _legacy._plot_lag_metrics(lag_metrics, plots_dir)


def _plot_backtest_scenario(predictions: pd.DataFrame, plots_dir: str):
    return _legacy._plot_backtest_scenario(predictions, plots_dir)


def _plot_backtest_timeline(folds: pd.DataFrame, output_path: str):
    return _legacy._plot_backtest_timeline(folds, output_path)


def save_forecast_plots(
    daily_df: pd.DataFrame,
    forecast: pd.DataFrame,
    champion_model: str,
    lag_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    plots_dir: str,
):
    return _legacy.save_forecast_plots(daily_df, forecast, champion_model, lag_metrics, predictions, plots_dir)


def run_backtest_and_save(
    daily_df: pd.DataFrame,
    paths: dict,
    horizon: int = DEFAULT_HORIZON,
    scenario_lags: Optional[Iterable[int]] = None,
    models: Optional[Iterable[str]] = None,
):
    _sync_facade_overrides()
    return _legacy.run_backtest_and_save(daily_df, paths, horizon, scenario_lags, models)


def run_forecast_and_save(daily_df: pd.DataFrame, paths: dict, horizon: int = DEFAULT_HORIZON):
    _sync_facade_overrides()
    return _legacy.run_forecast_and_save(daily_df, paths, horizon)


def run_backtest(daily_df: pd.DataFrame, models: Optional[Iterable[str]] = None, horizons=(7, 14, 30), cutoffs="rolling"):
    _sync_facade_overrides()
    return _legacy.run_backtest(daily_df, models, horizons, cutoffs)


def save_forecast_artifacts(
    daily_df: pd.DataFrame,
    forecast_path: str,
    metrics_path: str,
    comparison_path: str,
    plots_dir: Optional[str] = None,
):
    _sync_facade_overrides()
    return _legacy.save_forecast_artifacts(daily_df, forecast_path, metrics_path, comparison_path, plots_dir)


_ORIGINAL_RUN_BORUTA = _legacy._run_boruta
_ORIGINAL_SELECT_RECURSIVE_SCHEMA = _legacy._select_recursive_schema
_ORIGINAL_SELECT_CHAIN_SCHEMA = _legacy._select_chain_schema
_ORIGINAL_PREDICT_MODEL = _legacy.predict_model
_RUN_BORUTA_WRAPPER = _run_boruta
_SELECT_RECURSIVE_SCHEMA_WRAPPER = _select_recursive_schema
_SELECT_CHAIN_SCHEMA_WRAPPER = _select_chain_schema
_PREDICT_MODEL_WRAPPER = predict_model


__all__ = [
    "ArtifactStore",
    "BorutaFeatureSelector",
    "ChainMLAlgorithm",
    "FeatureSelectionConfig",
    "ForecastAlgorithm",
    "ForecastBacktester",
    "ForecastChampion",
    "ForecastEngine",
    "ForecastHyperparameterTuner",
    "ForecastModelRegistry",
    "ForecastPlotter",
    "ForecastPrediction",
    "ForecastRunConfig",
    "HyperparameterTuningConfig",
    "ModelCompetitionConfig",
    "RecursiveMLAlgorithm",
    "StatisticalAlgorithm",
    "algorithm_for_model",
    "available_models",
    "calculate_forecast_metrics",
    "forecast_demand",
    "load_champion",
    "predict_model",
    "run_backtest",
    "run_backtest_and_save",
    "run_backtest_detailed",
    "run_forecast_and_save",
    "save_champion",
    "save_forecast_artifacts",
    "save_forecast_plots",
    "select_champion",
]
