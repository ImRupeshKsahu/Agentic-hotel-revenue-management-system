import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from project_core.config import FORECAST_FORCE_KEEP_FEATURES
from forecasting_core.feature_engineering import FeatureEngineer
from forecasting_core.hyperparameter_tuning import (
    ForecastHyperparameterTuner,
    HyperparameterTuningConfig,
    TUNING_MAE_TIE_THRESHOLD_PP,
    load_tuning_payload,
    save_tuning_payload,
    tuned_params_from_payload,
)

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.multioutput import RegressorChain
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    ExtraTreesRegressor = None
    RandomForestRegressor = None
    Ridge = None
    ElasticNet = None
    RegressorChain = None
    make_pipeline = None
    StandardScaler = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from boruta import BorutaPy
except Exception:
    BorutaPy = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    ExponentialSmoothing = None
    SARIMAX = None


DEFAULT_HORIZON = 30
DEFAULT_SCENARIO_LAGS = [10, 14, 21, 30, 32, 45, 60, 90, 120]
DEFAULT_BACKTEST_STEP_DAYS = 7
DEFAULT_MIN_TRAIN_DAYS = 365
DEFAULT_AUDIT_FOLDS = 8
DEFAULT_INTERVAL_LEVEL = 0.90
DEFAULT_AUDIT_DRIFT_THRESHOLD = 0.25
DEFAULT_HYPERPARAM_TRIALS = 5
DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS = 5
DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP = TUNING_MAE_TIE_THRESHOLD_PP
MODEL_SELECTION_OBJECTIVE = "mae_pp_with_rmse_guardrail"
MODEL_SELECTION_MAE_TIE_THRESHOLD_PP = 0.50
BASELINE_PROFILE = "statistical"
ENHANCED_PROFILE = "boruta_selected"
LEGACY_BASELINE_PROFILE = "baseline"
LEGACY_ENHANCED_PROFILE = "enhanced_v1"
BORUTA_FEATURE_PROFILES = {ENHANCED_PROFILE, LEGACY_ENHANCED_PROFILE}
DEFAULT_FEATURE_PROFILES = [BASELINE_PROFILE, ENHANCED_PROFILE]
CHAIN_BORUTA_ANCHORS = [1, 14, 30]
CHAIN_BORUTA_MIN_ANCHORS = 2
BORUTA_MAX_ITER = 10
BORUTA_TREE_COUNT = "auto"
BORUTA_PERC = 80
TUNED_MODEL_PARAMS: dict[str, dict] = {}

STATISTICAL_MODELS = [
    "naive",
    "seasonal_naive_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "ewma_14",
    "ets",
    "sarimax",
]

DEFAULT_STATISTICAL_MODELS = [
    "seasonal_naive_7",
    "ewma_14",
    "sarimax",
]

ML_RECURSIVE_MODELS = [
    "random_forest_recursive",
    "extra_trees_recursive",
    "xgboost_recursive",
]

ML_CHAIN_MODELS = [
    "random_forest_chain",
    "extra_trees_chain",
    "xgboost_chain",
]

EXPERIMENTAL_MODELS = [
    "naive",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "ets",
    "ridge_chain",
    "elasticnet_chain",
]

SUPPORTED_MODELS = STATISTICAL_MODELS + ML_RECURSIVE_MODELS + ML_CHAIN_MODELS + [
    "ridge_chain",
    "elasticnet_chain",
]

DEFAULT_MODELS = DEFAULT_STATISTICAL_MODELS + ML_RECURSIVE_MODELS + ML_CHAIN_MODELS

MODEL_COMPLEXITY = {
    "naive": 1,
    "seasonal_naive_7": 1,
    "rolling_mean_7": 1,
    "rolling_mean_14": 1,
    "rolling_mean_28": 1,
    "ewma_14": 2,
    "ets": 3,
    "sarimax": 4,
    "ridge_chain": 3,
    "elasticnet_chain": 3,
    "random_forest_recursive": 4,
    "extra_trees_recursive": 4,
    "xgboost_recursive": 5,
    "random_forest_chain": 5,
    "extra_trees_chain": 5,
    "xgboost_chain": 6,
}

SARIMAX_CANDIDATES = [
    {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 7), "trend": "c"},
    {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 7), "trend": "n"},
    {"order": (2, 0, 1), "seasonal_order": (1, 0, 1, 7), "trend": "c"},
]


def _is_statistical_model(model_name: str) -> bool:
    return model_name in STATISTICAL_MODELS


def _is_boruta_feature_profile(feature_profile: str) -> bool:
    return feature_profile in BORUTA_FEATURE_PROFILES


def _uses_boruta_features(model_name: str, feature_profile: Optional[str] = None) -> bool:
    if _is_statistical_model(model_name):
        return False
    return feature_profile is None or _is_boruta_feature_profile(feature_profile)


def _normalize_feature_profile(model_name: str, feature_profile: Optional[str]) -> str:
    if _is_statistical_model(model_name):
        return BASELINE_PROFILE
    if feature_profile in BORUTA_FEATURE_PROFILES or feature_profile in {None, LEGACY_BASELINE_PROFILE, BASELINE_PROFILE}:
        return ENHANCED_PROFILE
    return str(feature_profile)


def _unavailable_model_reason(model_name: str) -> Optional[str]:
    """Return a human-readable reason when a requested model cannot run natively."""
    if model_name == "ets" and ExponentialSmoothing is None:
        return "statsmodels ExponentialSmoothing is unavailable"
    if model_name == "sarimax" and SARIMAX is None:
        return "statsmodels SARIMAX is unavailable"
    if model_name.startswith(("random_forest", "extra_trees", "ridge", "elasticnet")) and _base_estimator(model_name) is None:
        return "scikit-learn estimator is unavailable"
    if model_name.startswith("xgboost") and _base_estimator(model_name) is None:
        return "xgboost estimator is unavailable"
    if not _is_statistical_model(model_name) and not _boruta_available():
        return _boruta_unavailable_reason()
    return None


def available_models(models: Iterable[str]) -> list[str]:
    """Filter a model list down to those that can run without substituting another model."""
    return [model_name for model_name in models if _unavailable_model_reason(model_name) is None]


@dataclass(frozen=True)
class ForecastChampion:
    model: str
    strategy: str
    horizon: int
    selected_at: str
    metrics: dict
    feature_schema: list[str]
    feature_profile: str = BASELINE_PROFILE
    selected_historical_features: list[str] = field(default_factory=list)
    mandatory_features: list[str] = field(default_factory=list)
    feature_selection_metadata: dict = field(default_factory=dict)
    backtest_cadence_days: int = 7
    selection_objective: str = MODEL_SELECTION_OBJECTIVE
    mae_tie_threshold_pp: float = MODEL_SELECTION_MAE_TIE_THRESHOLD_PP
    interval_level: float = DEFAULT_INTERVAL_LEVEL
    interval_quantiles: dict = field(default_factory=dict)
    backtest_metadata: dict = field(default_factory=dict)
    hyperparameter_tuning_metadata: dict = field(default_factory=dict)


def _actuals(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Occupancy_Rate"].notna()].sort_values("Date").reset_index(drop=True)
    for col, default in {
        "Is_Weekend": 0,
        "Local_Event": 0,
        "Competitor_Rate": np.nan,
        "Booking_Pace": np.nan,
        "Cancellations": 0,
        "Bookings_Created": 0,
    }.items():
        if col not in df.columns:
            df[col] = default
    # Use only past information when filling historical covariates. Backfilling from
    # later rows would let earlier training windows borrow knowledge from the future.
    df["Competitor_Rate"] = df["Competitor_Rate"].ffill().fillna(120.0)
    df["Booking_Pace"] = df["Booking_Pace"].fillna(0)
    return df


def calculate_forecast_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {}

    error = predicted - actual
    non_zero = np.where(np.abs(actual) < 1e-6, np.nan, actual)
    mape = np.nanmean(np.abs(error / non_zero)) * 100
    smape = np.nanmean(2 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(predicted), 1e-6)) * 100
    wape = np.sum(np.abs(error)) / max(np.sum(np.abs(actual)), 1e-6) * 100
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAE_pp": float(mae * 100),
        "RMSE_pp": float(rmse * 100),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "WAPE": float(wape),
        "Bias": float(bias),
        "Bias_pp": float(bias * 100),
        "Abs_Bias_pp": float(abs(bias * 100)),
        "Accuracy": float(max(0, 100 - wape)),
        "Volatility": float(np.std(predicted)),
        "Stability": float(1 / (1 + np.std(error))),
    }


def _strategy_for_model(model_name: str) -> str:
    if model_name.endswith("_chain"):
        return "regressor_chain"
    if model_name.endswith("_recursive"):
        return "recursive_ml"
    return "statistical"


def _future_dates(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    start = history["Date"].max() + pd.Timedelta(days=1)
    future = pd.DataFrame({"Date": pd.date_range(start, periods=horizon, freq="D")})
    future["Is_Weekend"] = future["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    future["Local_Event"] = 0
    competitor_rate = history["Competitor_Rate"].dropna().tail(28).mean()
    future["Competitor_Rate"] = 120.0 if pd.isna(competitor_rate) else competitor_rate
    future["Booking_Pace"] = 0
    future["Cancellations"] = 0
    future["Bookings_Created"] = 0
    return future


_FEATURE_ENGINEER = FeatureEngineer(future_factory=_future_dates)


def _sarimax_exog(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    local_event = pd.to_numeric(df.get("Local_Event", 0), errors="coerce").fillna(0)
    is_weekend = dates.dt.dayofweek.isin([5, 6]).astype(int)

    exog = pd.DataFrame(
        {
            "is_weekend": is_weekend.astype(float),
            "dow_sin": np.sin(2 * np.pi * dates.dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * dates.dt.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dates.dt.month / 12),
            "local_event": local_event.astype(float),
        },
        index=df.index,
    )
    return exog.replace([np.inf, -np.inf], np.nan).fillna(0)


def _calendar_features(date: pd.Timestamp, prefix: str) -> dict:
    return _FEATURE_ENGINEER.calendar_features(date, prefix)


def _series_value_or_mean(series: pd.Series, lag: int) -> float:
    return _FEATURE_ENGINEER._series_value_or_mean(series, lag)


def _rolling_slope(series: pd.Series, window: int) -> float:
    return _FEATURE_ENGINEER._rolling_slope(series.tail(window))


def _seasonal_index(history: pd.DataFrame, group: str, value) -> float:
    return _FEATURE_ENGINEER._seasonal_index(history, group, value)


def _mandatory_feature_names(columns: Iterable[str]) -> list[str]:
    return _FEATURE_ENGINEER.mandatory_feature_names(columns)


def _split_feature_schema(feature_schema: Iterable[str]) -> tuple[list[str], list[str]]:
    return _FEATURE_ENGINEER.split_feature_schema(feature_schema)


def _baseline_feature_vector(history: pd.DataFrame, future: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    return _FEATURE_ENGINEER.feature_vector(history, future, horizon=horizon, feature_profile=BASELINE_PROFILE)


def _enhanced_feature_vector(history: pd.DataFrame, future: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    return _FEATURE_ENGINEER.feature_vector(history, future, horizon=horizon, feature_profile=ENHANCED_PROFILE)


def _feature_vector(
    history: pd.DataFrame,
    future: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    feature_profile: str = BASELINE_PROFILE,
) -> dict:
    return _FEATURE_ENGINEER.feature_vector(history, future, horizon=horizon, feature_profile=feature_profile)


def _build_chain_training(
    history: pd.DataFrame,
    horizon: int,
    min_history: int = 90,
    feature_profile: str = BASELINE_PROFILE,
):
    rows = []
    targets = []
    max_origin = len(history) - horizon - 1
    for origin_idx in range(min_history, max_origin + 1):
        origin_history = history.iloc[: origin_idx + 1].copy()
        realized_future = history.iloc[origin_idx + 1 : origin_idx + 1 + horizon].copy()
        planned_future = _future_dates(origin_history, horizon)
        rows.append(_feature_vector(origin_history, planned_future, horizon=horizon, feature_profile=feature_profile))
        targets.append(realized_future["Occupancy_Rate"].to_numpy())
    if not rows:
        return pd.DataFrame(), np.empty((0, horizon)), []
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    return x, np.vstack(targets), list(x.columns)


def _build_recursive_training(
    history: pd.DataFrame,
    min_history: int = 90,
    feature_profile: str = BASELINE_PROFILE,
):
    rows = []
    targets = []
    for target_idx in range(min_history + 1, len(history)):
        origin_history = history.iloc[:target_idx].copy()
        target_row = history.iloc[[target_idx]].copy()
        planned_future = _future_dates(origin_history, 1)
        rows.append(_feature_vector(origin_history, planned_future, horizon=1, feature_profile=feature_profile))
        targets.append(float(target_row["Occupancy_Rate"].iloc[0]))
    if not rows:
        return pd.DataFrame(), np.array([]), []
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    return x, np.array(targets), list(x.columns)


def _boruta_available() -> bool:
    return BorutaPy is not None and RandomForestRegressor is not None


def _boruta_unavailable_reason() -> Optional[str]:
    if BorutaPy is None:
        return "boruta BorutaPy is unavailable"
    if RandomForestRegressor is None:
        return "scikit-learn RandomForestRegressor is unavailable"
    return None


def _run_boruta(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    anchor: str,
) -> pd.DataFrame:
    force_keep = set(_force_keep_features(x_train.columns))
    historical_cols = [
        col
        for col in x_train.columns
        if col not in _mandatory_feature_names(x_train.columns) and col not in force_keep
    ]
    if not historical_cols:
        return pd.DataFrame(columns=["Feature", "Anchor", "Support", "Support_Weak", "Rank", "Force_Kept"])
    if not _boruta_available():
        raise RuntimeError(_boruta_unavailable_reason())

    selector_estimator = RandomForestRegressor(
        n_estimators=100 if BORUTA_TREE_COUNT == "auto" else BORUTA_TREE_COUNT,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=1,
    )
    selector = BorutaPy(
        estimator=selector_estimator,
        n_estimators=BORUTA_TREE_COUNT,
        perc=BORUTA_PERC,
        max_iter=BORUTA_MAX_ITER,
        random_state=42,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
        selector.fit(x_train[historical_cols].to_numpy(dtype=float), np.asarray(y_train, dtype=float))
    return pd.DataFrame(
        {
            "Feature": historical_cols,
            "Anchor": anchor,
            "Support": selector.support_.astype(bool),
            "Support_Weak": selector.support_weak_.astype(bool),
            "Rank": selector.ranking_.astype(int),
            "Force_Kept": False,
        }
    )


def _force_keep_features(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [feature for feature in FORECAST_FORCE_KEEP_FEATURES if feature in available]


def _force_keep_report(features: Iterable[str], anchor: str, strategy: str) -> pd.DataFrame:
    features = list(features)
    if not features:
        return pd.DataFrame(columns=["Feature", "Anchor", "Support", "Support_Weak", "Rank", "Force_Kept", "Strategy"])
    return pd.DataFrame(
        {
            "Feature": features,
            "Anchor": anchor,
            "Support": True,
            "Support_Weak": False,
            "Rank": 0,
            "Force_Kept": True,
            "Strategy": strategy,
        }
    )


def _stable_features_from_boruta(report: pd.DataFrame, min_anchor_count: int) -> tuple[list[str], str]:
    if report.empty:
        return [], "no_candidates"
    if "Force_Kept" in report.columns:
        report = report[~report["Force_Kept"].fillna(False)].copy()
        if report.empty:
            return [], "forced_only"
    strong = report.groupby("Feature")["Support"].sum()
    selected = strong[strong.ge(min_anchor_count)].index.tolist()
    if selected:
        return sorted(selected), "strong_support"

    weak = report.assign(_kept=report["Support"] | report["Support_Weak"]).groupby("Feature")["_kept"].sum()
    selected = weak[weak.ge(min_anchor_count)].index.tolist()
    if selected:
        return sorted(selected), "weak_support_fallback"

    avg_rank = report.groupby("Feature")["Rank"].mean().sort_values()
    return avg_rank.head(min(10, len(avg_rank))).index.tolist(), "rank_fallback"


def _select_recursive_schema(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[list[str], pd.DataFrame, dict]:
    mandatory = _mandatory_feature_names(x_train.columns)
    force_keep = _force_keep_features(x_train.columns)
    report = _run_boruta(x_train, y_train, anchor="recursive")
    selected_by_boruta, status = _stable_features_from_boruta(report, min_anchor_count=1)
    if force_keep and status == "rank_fallback":
        selected_by_boruta = []
        status = "force_keep_only_after_boruta_rejection"
    selected_historical = list(dict.fromkeys(force_keep + selected_by_boruta))
    schema = selected_historical + mandatory
    report = report.assign(
        Strategy="recursive_ml",
        Selection_Frequency=report["Support"].astype(int),
        Selected=report["Feature"].isin(selected_historical),
        Selection_Status=status,
    )
    forced_report = _force_keep_report(force_keep, anchor="recursive", strategy="recursive_ml").assign(
        Selection_Frequency=1,
        Selected=True,
        Selection_Status="force_kept",
    )
    report = pd.concat([forced_report, report], ignore_index=True)
    return schema, report, {
        "selected_historical_features": selected_historical,
        "boruta_selected_features": selected_by_boruta,
        "force_kept_features": force_keep,
        "mandatory_features": mandatory,
        "selection_status": status,
        "anchors": ["recursive"],
    }


def _select_chain_schema(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    anchors: Iterable[int] = CHAIN_BORUTA_ANCHORS,
) -> tuple[list[str], pd.DataFrame, dict]:
    mandatory = _mandatory_feature_names(x_train.columns)
    force_keep = _force_keep_features(x_train.columns)
    reports = []
    valid_anchors = []
    for anchor in anchors:
        if 1 <= int(anchor) <= y_train.shape[1]:
            valid_anchors.append(int(anchor))
            reports.append(_run_boruta(x_train, y_train[:, int(anchor) - 1], anchor=f"h{int(anchor)}"))
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    min_anchor_count = min(CHAIN_BORUTA_MIN_ANCHORS, len(valid_anchors)) if valid_anchors else 0
    selected_by_boruta, status = _stable_features_from_boruta(report, min_anchor_count=min_anchor_count)
    if force_keep and status == "rank_fallback":
        selected_by_boruta = []
        status = "force_keep_only_after_boruta_rejection"
    frequency = report.groupby("Feature")["Support"].sum() if not report.empty else pd.Series(dtype=float)
    selected_historical = list(dict.fromkeys(force_keep + selected_by_boruta))
    schema = selected_historical + mandatory
    if not report.empty:
        report = report.assign(
            Strategy="regressor_chain",
            Selection_Frequency=report["Feature"].map(frequency).fillna(0).astype(int),
            Selected=report["Feature"].isin(selected_historical),
            Selection_Status=status,
        )
    forced_reports = [
        _force_keep_report(force_keep, anchor=f"h{anchor}", strategy="regressor_chain")
        for anchor in valid_anchors
    ]
    forced_report = pd.concat(forced_reports, ignore_index=True) if forced_reports else pd.DataFrame()
    if not forced_report.empty:
        forced_report = forced_report.assign(
            Selection_Frequency=len(valid_anchors),
            Selected=True,
            Selection_Status="force_kept",
        )
        report = pd.concat([forced_report, report], ignore_index=True)
    return schema, report, {
        "selected_historical_features": selected_historical,
        "boruta_selected_features": selected_by_boruta,
        "force_kept_features": force_keep,
        "mandatory_features": mandatory,
        "selection_status": status,
        "anchors": [f"h{anchor}" for anchor in valid_anchors],
        "min_anchor_count": min_anchor_count,
    }


def _select_production_feature_schemas(
    history: pd.DataFrame,
    horizon: int,
    model_specs: Iterable[dict],
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    """Select Boruta schemas once from all currently available history."""
    specs = list(model_specs)
    if not any(_uses_boruta_features(spec["model"], spec.get("feature_profile")) for spec in specs):
        return {}, pd.DataFrame(), {}

    selected_at = pd.Timestamp.now().isoformat()
    schemas: dict[str, list[str]] = {}
    metadata: dict[str, dict] = {}
    reports = []
    max_date = pd.to_datetime(history["Date"].max())

    if any(spec["model"].endswith("_recursive") for spec in specs):
        x_recursive, y_recursive, _ = _build_recursive_training(history, feature_profile=ENHANCED_PROFILE)
        recursive_schema, recursive_report, recursive_metadata = _select_recursive_schema(x_recursive, y_recursive)
        schemas["recursive_ml"] = recursive_schema
        metadata["recursive_ml"] = {**recursive_metadata, "selected_at": selected_at}
        if not recursive_report.empty:
            reports.append(
                recursive_report.assign(
                    Fold_ID="production_schema",
                    Split="production_schema",
                    Cutoff=max_date,
                    Feature_Profile=ENHANCED_PROFILE,
                    Selection_Mode="one_time_full_history",
                    Selected_At=selected_at,
                )
            )

    if any(spec["model"].endswith("_chain") for spec in specs):
        x_chain, y_chain, _ = _build_chain_training(history, horizon=horizon, feature_profile=ENHANCED_PROFILE)
        chain_schema, chain_report, chain_metadata = _select_chain_schema(x_chain, y_chain)
        schemas["regressor_chain"] = chain_schema
        metadata["regressor_chain"] = {**chain_metadata, "selected_at": selected_at}
        if not chain_report.empty:
            reports.append(
                chain_report.assign(
                    Fold_ID="production_schema",
                    Split="production_schema",
                    Cutoff=max_date,
                    Feature_Profile=ENHANCED_PROFILE,
                    Selection_Mode="one_time_full_history",
                    Selected_At=selected_at,
                )
            )

    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    return schemas, report, metadata


def _model_tuned_params(model_name: str, tuned_params: Optional[dict] = None) -> dict:
    if tuned_params is not None:
        return dict(tuned_params)
    return dict(TUNED_MODEL_PARAMS.get(model_name, {}))


def _base_estimator(model_name: str, tuned_params: Optional[dict] = None):
    tuned_params = _model_tuned_params(model_name, tuned_params)
    if model_name.startswith("ridge"):
        if Ridge is None:
            return None
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    if model_name.startswith("elasticnet"):
        if ElasticNet is None:
            return None
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.005, l1_ratio=0.2, max_iter=5000, random_state=42))
    if model_name.startswith("random_forest"):
        if RandomForestRegressor is None:
            return None
        params = {"n_estimators": 18, "min_samples_leaf": 6, "random_state": 42, "n_jobs": -1}
        params.update(tuned_params)
        params["random_state"] = 42
        params["n_jobs"] = -1
        return RandomForestRegressor(**params)
    if model_name.startswith("extra_trees"):
        if ExtraTreesRegressor is None:
            return None
        params = {"n_estimators": 24, "min_samples_leaf": 5, "random_state": 42, "n_jobs": -1}
        params.update(tuned_params)
        params["random_state"] = 42
        params["n_jobs"] = -1
        return ExtraTreesRegressor(**params)
    if model_name.startswith("xgboost"):
        if XGBRegressor is None:
            return None
        params = {
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
        }
        params.update(tuned_params)
        params["objective"] = "reg:squarederror"
        params["random_state"] = 42
        params["n_jobs"] = 1
        return XGBRegressor(**params)
    return None


def _predict_statistical(history: pd.DataFrame, horizon: int, model_name: str) -> np.ndarray:
    y = history["Occupancy_Rate"].astype(float).to_numpy()
    if len(y) == 0:
        return np.zeros(horizon)
    if model_name == "naive":
        return np.repeat(y[-1], horizon)
    if model_name == "seasonal_naive_7":
        base = y[-7:] if len(y) >= 7 else y
        return np.resize(base, horizon)
    if model_name.startswith("rolling_mean_"):
        window = int(model_name.rsplit("_", 1)[-1])
        return np.repeat(np.mean(y[-window:]), horizon)
    if model_name == "ewma_14":
        return np.repeat(pd.Series(y).ewm(span=14, adjust=False).mean().iloc[-1], horizon)
    if model_name == "ets" and ExponentialSmoothing is not None and len(y) >= 60:
        try:
            model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
            fitted = model.fit(optimized=True)
            return np.asarray(fitted.forecast(horizon), dtype=float)
        except Exception:
            return _predict_statistical(history, horizon, "seasonal_naive_7")
    if model_name == "sarimax":
        if SARIMAX is None or len(y) < 120:
            return _predict_statistical(history, horizon, "seasonal_naive_7")
        try:
            exog = _sarimax_exog(history)
            future_exog = _sarimax_exog(_future_dates(history, horizon))
            best_fit = None
            best_aic = np.inf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for candidate in SARIMAX_CANDIDATES:
                    model = SARIMAX(
                        y,
                        exog=exog,
                        order=candidate["order"],
                        seasonal_order=candidate["seasonal_order"],
                        trend=candidate["trend"],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False, maxiter=100)
                    fit_aic = float(fitted.aic) if np.isfinite(fitted.aic) else np.inf
                    if fit_aic < best_aic:
                        best_fit = fitted
                        best_aic = fit_aic
            if best_fit is None:
                return _predict_statistical(history, horizon, "seasonal_naive_7")
            return np.asarray(best_fit.forecast(horizon, exog=future_exog), dtype=float)
        except Exception:
            return _predict_statistical(history, horizon, "seasonal_naive_7")
    return _predict_statistical(history, horizon, "rolling_mean_7")


def _predict_chain(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    if RegressorChain is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    estimator = _base_estimator(model_name, tuned_params=tuned_params)
    if estimator is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    x_train, y_train, feature_schema = _build_chain_training(history, horizon=horizon, feature_profile=feature_profile)
    if len(x_train) < 30:
        return _predict_statistical(history, horizon, "rolling_mean_7"), feature_schema
    if selected_schema:
        feature_schema = selected_schema
    model = RegressorChain(estimator, order=list(range(horizon)))
    model.fit(x_train.reindex(columns=feature_schema).fillna(0), y_train)
    future = _future_dates(history, horizon)
    x_future = pd.DataFrame(
        [_feature_vector(history, future, horizon=horizon, feature_profile=feature_profile)]
    ).reindex(columns=feature_schema).fillna(0)
    return np.asarray(model.predict(x_future)[0], dtype=float), feature_schema


def _predict_recursive(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    estimator = _base_estimator(model_name, tuned_params=tuned_params)
    if estimator is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    x_train, y_train, feature_schema = _build_recursive_training(history, feature_profile=feature_profile)
    if len(x_train) < 30:
        return _predict_statistical(history, horizon, "rolling_mean_7"), feature_schema
    if selected_schema:
        feature_schema = selected_schema
    estimator.fit(x_train.reindex(columns=feature_schema).fillna(0), y_train)

    recursive = history.copy()
    preds = []
    for step in range(horizon):
        future = _future_dates(recursive, 1)
        x_future = pd.DataFrame(
            [_feature_vector(recursive, future, horizon=1, feature_profile=feature_profile)]
        ).reindex(columns=feature_schema).fillna(0)
        pred = float(estimator.predict(x_future)[0])
        pred = float(np.clip(pred, 0, 1))
        preds.append(pred)
        new_row = future.iloc[[0]].copy()
        new_row["Occupancy_Rate"] = pred
        for col in recursive.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        recursive = pd.concat([recursive, new_row[recursive.columns]], ignore_index=True)
    return np.asarray(preds, dtype=float), feature_schema


def predict_model(
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    selected_schema: Optional[list[str]] = None,
    tuned_params: Optional[dict] = None,
):
    if model_name in STATISTICAL_MODELS:
        return np.clip(_predict_statistical(history, horizon, model_name), 0, 1), []
    if model_name.endswith("_chain"):
        preds, schema = _predict_chain(
            history,
            horizon,
            model_name,
            feature_profile=feature_profile,
            selected_schema=selected_schema,
            tuned_params=tuned_params,
        )
        return np.clip(preds, 0, 1), schema
    if model_name.endswith("_recursive"):
        preds, schema = _predict_recursive(
            history,
            horizon,
            model_name,
            feature_profile=feature_profile,
            selected_schema=selected_schema,
            tuned_params=tuned_params,
        )
        return np.clip(preds, 0, 1), schema
    return np.clip(_predict_statistical(history, horizon, "rolling_mean_7"), 0, 1), []


def _generate_weekly_folds(
    df: pd.DataFrame,
    horizon: int,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    step_days: int = DEFAULT_BACKTEST_STEP_DAYS,
    audit_folds: int = DEFAULT_AUDIT_FOLDS,
) -> pd.DataFrame:
    max_date = pd.to_datetime(df["Date"].max())
    min_date = pd.to_datetime(df["Date"].min())
    last_full_cutoff = max_date - pd.Timedelta(days=horizon)
    earliest_cutoff = min_date + pd.Timedelta(days=min_train_days)

    cutoffs = []
    cutoff = last_full_cutoff
    while cutoff >= earliest_cutoff:
        actual_future = df[(df["Date"].gt(cutoff)) & (df["Date"].le(cutoff + pd.Timedelta(days=horizon)))]
        if len(actual_future) == horizon:
            cutoffs.append(cutoff)
        cutoff -= pd.Timedelta(days=step_days)
    cutoffs = list(reversed(cutoffs))

    if not cutoffs:
        return pd.DataFrame(
            columns=[
                "Fold_ID",
                "Cutoff",
                "Split",
                "Train_Start",
                "Train_End",
                "Validation_Start",
                "Validation_End",
            ]
        )

    effective_audit_folds = min(max(int(audit_folds), 0), max(len(cutoffs) - 1, 0))
    rows = []
    for idx, cutoff in enumerate(cutoffs, start=1):
        split = "audit" if idx > len(cutoffs) - effective_audit_folds else "selection"
        rows.append(
            {
                "Fold_ID": f"fold_{idx:03d}",
                "Cutoff": cutoff,
                "Split": split,
                "Train_Start": min_date,
                "Train_End": cutoff,
                "Validation_Start": cutoff + pd.Timedelta(days=1),
                "Validation_End": cutoff + pd.Timedelta(days=horizon),
            }
        )
    return pd.DataFrame(rows)


def _scenario_folds(df: pd.DataFrame, scenario_lags: Iterable[int], horizon: int) -> pd.DataFrame:
    max_date = df["Date"].max()
    min_date = df["Date"].min()
    rows = []
    for idx, lag in enumerate(scenario_lags, start=1):
        cutoff = max_date - pd.Timedelta(days=int(lag))
        if cutoff <= min_date + pd.Timedelta(days=120):
            continue
        actual_future = df[(df["Date"].gt(cutoff)) & (df["Date"].le(cutoff + pd.Timedelta(days=horizon)))]
        if len(actual_future) != horizon:
            continue
        rows.append(
            {
                "Fold_ID": f"scenario_{idx:03d}",
                "Cutoff": cutoff,
                "Split": "selection",
                "Train_Start": min_date,
                "Train_End": cutoff,
                "Validation_Start": cutoff + pd.Timedelta(days=1),
                "Validation_End": cutoff + pd.Timedelta(days=horizon),
            }
        )
    return pd.DataFrame(rows)


def _metrics_from_predictions(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for key, group in predictions.groupby(group_cols):
        metrics = calculate_forecast_metrics(group["Actual"].to_numpy(), group["Predicted"].to_numpy())
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(group_cols, key)), "Observations": len(group), **metrics})
    return pd.DataFrame(rows)


def _with_feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    if "Feature_Profile" in df.columns:
        return df
    copy = df.copy()
    copy["Feature_Profile"] = BASELINE_PROFILE
    return copy


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = _with_feature_profile(fold_metrics)
    subset = subset[subset["Split"].eq(split)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset = _ensure_point_metrics(subset)
    metric_cols = [
        "MAE",
        "RMSE",
        "MAE_pp",
        "RMSE_pp",
        "MAPE",
        "sMAPE",
        "WAPE",
        "Bias",
        "Bias_pp",
        "Accuracy",
        "Volatility",
        "Stability",
    ]
    grouped = (
        subset.groupby(["Feature_Profile", "Model", "Strategy"], as_index=False)
        .agg(
            Folds=("Fold_ID", "nunique"),
            Observations=("Observations", "sum"),
            **{col: (col, "mean") for col in metric_cols},
        )
    )
    grouped["Abs_Bias"] = grouped["Bias"].abs()
    grouped["Abs_Bias_pp"] = grouped["Bias_pp"].abs()
    return grouped


def _ensure_point_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    copy = metrics.copy()
    if "MAE_pp" not in copy.columns and "MAE" in copy.columns:
        copy["MAE_pp"] = pd.to_numeric(copy["MAE"], errors="coerce") * 100
    if "MAE_pp" not in copy.columns and "WAPE" in copy.columns:
        copy["MAE_pp"] = pd.to_numeric(copy["WAPE"], errors="coerce")
    if "RMSE_pp" not in copy.columns and "RMSE" in copy.columns:
        copy["RMSE_pp"] = pd.to_numeric(copy["RMSE"], errors="coerce") * 100
    if "RMSE_pp" not in copy.columns and "MAE_pp" in copy.columns:
        copy["RMSE_pp"] = pd.to_numeric(copy["MAE_pp"], errors="coerce")
    if "Bias_pp" not in copy.columns and "Bias" in copy.columns:
        copy["Bias_pp"] = pd.to_numeric(copy["Bias"], errors="coerce") * 100
    if "Bias_pp" not in copy.columns:
        copy["Bias_pp"] = 0.0
    if "Abs_Bias_pp" not in copy.columns and "Bias_pp" in copy.columns:
        copy["Abs_Bias_pp"] = pd.to_numeric(copy["Bias_pp"], errors="coerce").abs()
    if "Abs_Bias" not in copy.columns and "Bias" in copy.columns:
        copy["Abs_Bias"] = pd.to_numeric(copy["Bias"], errors="coerce").abs()
    if "Complexity" not in copy.columns and "Model" in copy.columns:
        copy["Complexity"] = copy["Model"].map(MODEL_COMPLEXITY).fillna(5)
    return copy


def _sort_model_competition_metrics(
    metrics: pd.DataFrame,
    mae_tie_threshold_pp: float = MODEL_SELECTION_MAE_TIE_THRESHOLD_PP,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    copy = _ensure_point_metrics(metrics)
    best_mae = pd.to_numeric(copy["MAE_pp"], errors="coerce").min()
    if not np.isfinite(best_mae):
        return copy.sort_values(["Complexity", "Model"], ascending=[True, True])

    eligible_mask = copy["MAE_pp"].le(best_mae + float(mae_tie_threshold_pp))
    eligible = copy[eligible_mask].sort_values(
        ["RMSE_pp", "MAE_pp", "Abs_Bias_pp", "Complexity", "Model"],
        ascending=[True, True, True, True, True],
    )
    remaining = copy[~eligible_mask].sort_values(
        ["MAE_pp", "RMSE_pp", "Abs_Bias_pp", "Complexity", "Model"],
        ascending=[True, True, True, True, True],
    )
    return pd.concat([eligible, remaining], ignore_index=True)


def _export_model_metrics(metrics: pd.DataFrame, include_audit_columns: bool = False) -> pd.DataFrame:
    """Keep exported model/audit summaries focused on readable percent-scale metrics."""
    if metrics.empty:
        return metrics
    copy = _ensure_point_metrics(metrics)
    columns = [
        "Feature_Profile",
        "Model",
        "Strategy",
        "Folds",
        "Observations",
        "MAE_pp",
        "RMSE_pp",
        "Bias_pp",
        "Abs_Bias_pp",
        "MAPE",
        "WAPE",
        "Volatility",
        "Stability",
        "Complexity",
    ]
    if include_audit_columns:
        columns.extend(
            [
                "Is_Champion",
                "Selection_Mean_Fold_MAE_pp",
                "Audit_Drift_Ratio",
                "Audit_Status",
                "Interval_Coverage",
            ]
        )
    return copy[[col for col in columns if col in copy.columns]]


def _model_specs(models: Iterable[str]) -> list[dict]:
    specs = []
    for model_name in models:
        if model_name in STATISTICAL_MODELS:
            specs.append({"model": model_name, "feature_profile": BASELINE_PROFILE})
            continue
        if _boruta_available():
            specs.append({"model": model_name, "feature_profile": ENHANCED_PROFILE})
    return specs


def _feature_manifest(
    history: pd.DataFrame,
    horizon: int,
    champion_profile: str,
    champion_schema: Iterable[str],
) -> pd.DataFrame:
    future = _future_dates(history, horizon)
    champion_schema = set(champion_schema)
    rows = []
    profiles = [BASELINE_PROFILE] if champion_profile in {BASELINE_PROFILE, LEGACY_BASELINE_PROFILE} else [ENHANCED_PROFILE]
    for feature_profile in profiles:
        if feature_profile == ENHANCED_PROFILE and not _boruta_available():
            continue
        candidate_features = list(
            _feature_vector(
                history,
                future,
                horizon=horizon,
                feature_profile=feature_profile,
            ).keys()
        )
        mandatory = set(_mandatory_feature_names(candidate_features))
        force_keep = set(_force_keep_features(candidate_features))
        for feature in candidate_features:
            feature_role = "mandatory_future_known" if feature in mandatory else "historical_candidate"
            if feature in force_keep:
                feature_role = "force_kept_historical"
            rows.append(
                {
                    "Feature_Profile": feature_profile,
                    "Feature": feature,
                    "Feature_Role": feature_role,
                    "Is_Champion_Profile": feature_profile == champion_profile,
                    "Selected_In_Champion": feature_profile == champion_profile and feature in champion_schema,
                }
            )
    return pd.DataFrame(rows)


def _calibrate_interval_quantiles(
    predictions: pd.DataFrame,
    model_name: str,
    feature_profile: str = BASELINE_PROFILE,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> tuple[pd.DataFrame, dict]:
    predictions = _with_feature_profile(predictions)
    model_predictions = predictions[
        predictions["Model"].eq(model_name) & predictions["Feature_Profile"].eq(feature_profile)
    ].copy()
    if model_predictions.empty:
        return pd.DataFrame(), {}

    residual = model_predictions["Actual"] - model_predictions["Predicted"]
    model_predictions["Residual"] = residual
    alpha = max(0.0, min(1.0, 1 - interval_level))
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    quantiles = (
        model_predictions.groupby("Lag")["Residual"]
        .quantile([lower_q, upper_q])
        .unstack()
        .rename(columns={lower_q: "Lower_Residual", upper_q: "Upper_Residual"})
        .reset_index()
        .sort_values("Lag")
    )
    payload = {
        str(int(row.Lag)): {
            "lower_residual": float(row.Lower_Residual),
            "upper_residual": float(row.Upper_Residual),
        }
        for row in quantiles.itertuples(index=False)
    }
    return quantiles, payload


def _interval_bounds_for_lag(prediction: float, lag: int, interval_quantiles: dict) -> tuple[float, float]:
    quantile = interval_quantiles.get(str(int(lag))) or interval_quantiles.get(int(lag))
    if not quantile:
        return float("nan"), float("nan")
    lower = float(np.clip(prediction + float(quantile["lower_residual"]), 0, 1))
    upper = float(np.clip(prediction + float(quantile["upper_residual"]), 0, 1))
    return lower, upper


def _interval_coverage(
    predictions: pd.DataFrame,
    model_name: str,
    feature_profile: str,
    interval_quantiles: dict,
) -> pd.DataFrame:
    predictions = _with_feature_profile(predictions)
    model_predictions = predictions[
        predictions["Model"].eq(model_name) & predictions["Feature_Profile"].eq(feature_profile)
    ].copy()
    if model_predictions.empty or not interval_quantiles:
        return pd.DataFrame()
    bounds = model_predictions.apply(
        lambda row: _interval_bounds_for_lag(row["Predicted"], int(row["Lag"]), interval_quantiles),
        axis=1,
        result_type="expand",
    )
    model_predictions[["Lower_Bound", "Upper_Bound"]] = bounds
    model_predictions["Covered"] = (
        model_predictions["Actual"].ge(model_predictions["Lower_Bound"])
        & model_predictions["Actual"].le(model_predictions["Upper_Bound"])
    )
    coverage = model_predictions.groupby("Lag", as_index=False).agg(
        Observations=("Covered", "size"),
        Interval_Coverage=("Covered", "mean"),
    )
    return coverage


def _audit_status(
    selection_mean_metric: float,
    audit_mean_metric: float,
    drift_threshold: float = DEFAULT_AUDIT_DRIFT_THRESHOLD,
) -> tuple[float, str]:
    drift_ratio = (
        float(audit_mean_metric / selection_mean_metric)
        if np.isfinite(selection_mean_metric) and selection_mean_metric > 0 and np.isfinite(audit_mean_metric)
        else np.nan
    )
    status = (
        "recent_degradation_flagged"
        if np.isfinite(drift_ratio) and drift_ratio > 1 + drift_threshold
        else "ok"
    )
    return drift_ratio, status


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
    df = _actuals(daily_df)
    requested_models = list(models or DEFAULT_MODELS)
    unavailable_models = {
        model_name: reason
        for model_name in requested_models
        if (reason := _unavailable_model_reason(model_name)) is not None
    }
    models = [model_name for model_name in requested_models if model_name not in unavailable_models]
    for model_name, reason in unavailable_models.items():
        print(f"Skipping {model_name}: {reason}", flush=True)
    if any(model_name not in STATISTICAL_MODELS for model_name in models):
        print(
            "Feature selection enabled: evaluating one Boruta-selected profile per ML model.",
            flush=True,
        )
    model_specs = _model_specs(models)
    production_schemas, boruta_selection_report, production_schema_metadata = _select_production_feature_schemas(
        df,
        horizon,
        model_specs,
    )
    folds = (
        _scenario_folds(df, list(scenario_lags), horizon)
        if scenario_lags is not None
        else _generate_weekly_folds(df, horizon, min_train_days=min_train_days, step_days=step_days, audit_folds=audit_folds)
    )

    prediction_rows = []
    for fold in folds.itertuples(index=False):
        cutoff = pd.to_datetime(fold.Cutoff)
        train = df[df["Date"].le(cutoff)].copy()
        actual_future = df[(df["Date"].gt(cutoff)) & (df["Date"].le(cutoff + pd.Timedelta(days=horizon)))].copy()
        if len(actual_future) != horizon:
            continue
        print(
            f"Backtest fold {fold.Fold_ID}: train through {cutoff.date()} "
            f"({fold.Split}), validate {len(actual_future)} days",
            flush=True,
        )
        for spec in model_specs:
            model_name = spec["model"]
            feature_profile = spec["feature_profile"]
            strategy = _strategy_for_model(model_name)
            selected_schema = production_schemas.get(strategy) if _uses_boruta_features(model_name, feature_profile) else None
            print(f"  fitting {model_name} [{feature_profile}]", flush=True)
            preds, _ = predict_model(
                train,
                horizon,
                model_name,
                feature_profile=feature_profile,
                selected_schema=selected_schema,
            )
            valid_len = min(len(actual_future), len(preds))
            for lag_idx in range(valid_len):
                actual_row = actual_future.iloc[lag_idx]
                prediction_rows.append(
                    {
                        "Feature_Profile": feature_profile,
                        "Model": model_name,
                        "Strategy": strategy,
                        "Fold_ID": fold.Fold_ID,
                        "Split": fold.Split,
                        "Cutoff": cutoff,
                        "Lag": lag_idx + 1,
                        "Date": actual_row["Date"],
                        "Actual": float(actual_row["Occupancy_Rate"]),
                        "Predicted": float(preds[lag_idx]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = _metrics_from_predictions(
        predictions,
        ["Split", "Fold_ID", "Feature_Profile", "Model", "Strategy", "Cutoff"],
    )
    overall = _aggregate_fold_metrics(fold_metrics, split="selection")
    lag_metrics = _metrics_from_predictions(
        predictions[predictions["Split"].eq("selection")],
        ["Feature_Profile", "Model", "Strategy", "Lag"],
    )
    scenario_metrics = fold_metrics.copy()
    if not overall.empty:
        overall = _sort_model_competition_metrics(overall)
    if return_feature_artifacts:
        return (
            overall,
            lag_metrics,
            scenario_metrics,
            predictions,
            fold_metrics,
            folds,
            boruta_selection_report,
            production_schemas,
            production_schema_metadata,
        )
    return overall, lag_metrics, scenario_metrics, predictions, fold_metrics, folds


def select_champion(overall_metrics: pd.DataFrame, horizon: int, feature_schema: list[str] | None = None) -> ForecastChampion:
    if overall_metrics.empty:
        model_name = "seasonal_naive_7"
        feature_profile = BASELINE_PROFILE
        metrics = {}
    else:
        winner = _sort_model_competition_metrics(overall_metrics).iloc[0]
        model_name = winner["Model"]
        feature_profile = _normalize_feature_profile(model_name, winner.get("Feature_Profile", BASELINE_PROFILE))
        metrics = winner.drop(labels=[c for c in ["Abs_Bias", "Complexity"] if c in winner.index]).to_dict()
    return ForecastChampion(
        model=model_name,
        strategy=_strategy_for_model(model_name),
        horizon=horizon,
        selected_at=pd.Timestamp.now().isoformat(),
        metrics=metrics,
        feature_schema=feature_schema or [],
        feature_profile=feature_profile,
        selection_objective=MODEL_SELECTION_OBJECTIVE,
        mae_tie_threshold_pp=MODEL_SELECTION_MAE_TIE_THRESHOLD_PP,
    )


def _select_champion_with_acceptance(
    overall_metrics: pd.DataFrame,
    audit_summary: pd.DataFrame,
    horizon: int,
) -> tuple[ForecastChampion, dict]:
    if overall_metrics.empty:
        return select_champion(overall_metrics, horizon=horizon), {"acceptance_rule": "no_metrics"}

    ordered = _sort_model_competition_metrics(overall_metrics)
    tentative = ordered.iloc[0]
    tentative_profile = _normalize_feature_profile(tentative["Model"], tentative.get("Feature_Profile", BASELINE_PROFILE))
    best_mae_pp = float(pd.to_numeric(_ensure_point_metrics(overall_metrics)["MAE_pp"], errors="coerce").min())
    acceptance = {
        "acceptance_rule": MODEL_SELECTION_OBJECTIVE,
        "selection_objective": MODEL_SELECTION_OBJECTIVE,
        "mae_tie_threshold_pp": MODEL_SELECTION_MAE_TIE_THRESHOLD_PP,
        "best_mae_pp": best_mae_pp,
        "tentative_model": tentative["Model"],
        "tentative_feature_profile": tentative_profile,
        "tentative_mae_pp": float(tentative.get("MAE_pp", np.nan)),
        "tentative_rmse_pp": float(tentative.get("RMSE_pp", np.nan)),
    }
    return select_champion(overall_metrics, horizon=horizon), {
        **acceptance,
        "accepted": True,
        "reason": "single_profile_no_baseline_comparator",
    }


def save_champion(champion: ForecastChampion, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "model": champion.model,
                "strategy": champion.strategy,
                "horizon": champion.horizon,
                "selected_at": champion.selected_at,
                "metrics": champion.metrics,
                "feature_schema": champion.feature_schema,
                "feature_profile": champion.feature_profile,
                "selected_historical_features": champion.selected_historical_features,
                "mandatory_features": champion.mandatory_features,
                "feature_selection_metadata": champion.feature_selection_metadata,
                "backtest_cadence_days": champion.backtest_cadence_days,
                "selection_objective": champion.selection_objective,
                "mae_tie_threshold_pp": champion.mae_tie_threshold_pp,
                "interval_level": champion.interval_level,
                "interval_quantiles": champion.interval_quantiles,
                "backtest_metadata": champion.backtest_metadata,
                "hyperparameter_tuning_metadata": champion.hyperparameter_tuning_metadata,
            },
            f,
            indent=4,
            default=str,
        )


def _safe_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(fallback, index=index)
        print(f"Could not overwrite locked file {path}. Saved copy to {fallback}", flush=True)
        return fallback


def load_champion(path: str, default_horizon: int = DEFAULT_HORIZON) -> ForecastChampion:
    if not os.path.exists(path):
        return ForecastChampion(
            model="seasonal_naive_7",
            strategy="statistical",
            horizon=default_horizon,
            selected_at=pd.Timestamp.now().isoformat(),
            metrics={},
            feature_schema=[],
            feature_profile=BASELINE_PROFILE,
            selected_historical_features=[],
            mandatory_features=[],
            feature_selection_metadata={},
            interval_level=DEFAULT_INTERVAL_LEVEL,
            interval_quantiles={},
            backtest_metadata={},
            hyperparameter_tuning_metadata={},
        )
    with open(path, "r") as f:
        data = json.load(f)
    model_name = data.get("model", "seasonal_naive_7")
    return ForecastChampion(
        model=model_name,
        strategy=data.get("strategy", _strategy_for_model(model_name)),
        horizon=int(data.get("horizon", default_horizon)),
        selected_at=data.get("selected_at", pd.Timestamp.now().isoformat()),
        metrics=data.get("metrics", {}),
        feature_schema=data.get("feature_schema", []),
        feature_profile=_normalize_feature_profile(model_name, data.get("feature_profile", BASELINE_PROFILE)),
        selected_historical_features=data.get("selected_historical_features", []),
        mandatory_features=data.get("mandatory_features", []),
        feature_selection_metadata=data.get("feature_selection_metadata", {}),
        backtest_cadence_days=int(data.get("backtest_cadence_days", 7)),
        selection_objective=data.get("selection_objective", MODEL_SELECTION_OBJECTIVE),
        mae_tie_threshold_pp=float(data.get("mae_tie_threshold_pp", MODEL_SELECTION_MAE_TIE_THRESHOLD_PP)),
        interval_level=float(data.get("interval_level", DEFAULT_INTERVAL_LEVEL)),
        interval_quantiles=data.get("interval_quantiles", {}),
        backtest_metadata=data.get("backtest_metadata", {}),
        hyperparameter_tuning_metadata=data.get("hyperparameter_tuning_metadata", {}),
    )


def forecast_demand(
    daily_df: pd.DataFrame,
    selected_model: str = "seasonal_naive_7",
    horizon_days: int = DEFAULT_HORIZON,
    interval_quantiles: Optional[dict] = None,
    feature_profile: str = BASELINE_PROFILE,
    feature_schema: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    history = _actuals(daily_df)
    feature_profile = _normalize_feature_profile(selected_model, feature_profile)
    selected_schema = feature_schema
    if selected_schema is None and _uses_boruta_features(selected_model, feature_profile):
        if selected_model.endswith("_recursive"):
            x_train, y_train, _ = _build_recursive_training(history, feature_profile=feature_profile)
            selected_schema, _, _ = _select_recursive_schema(x_train, y_train)
        elif selected_model.endswith("_chain"):
            x_train, y_train, _ = _build_chain_training(history, horizon=horizon_days, feature_profile=feature_profile)
            selected_schema, _, _ = _select_chain_schema(x_train, y_train)
    preds, resolved_schema = predict_model(
        history,
        horizon_days,
        selected_model,
        feature_profile=feature_profile,
        selected_schema=selected_schema,
    )
    future = _future_dates(history, horizon_days)
    future["Forecasted_Occupancy"] = np.clip(preds, 0, 1)
    if interval_quantiles:
        bounds = [
            _interval_bounds_for_lag(float(pred), lag + 1, interval_quantiles)
            for lag, pred in enumerate(future["Forecasted_Occupancy"])
        ]
        future[["Min_Occupancy", "Max_Occupancy"]] = pd.DataFrame(bounds, index=future.index)
    else:
        residual_std = history["Occupancy_Rate"].diff().dropna().std()
        residual_std = 0.08 if pd.isna(residual_std) else float(residual_std)
        future["Min_Occupancy"] = (future["Forecasted_Occupancy"] - residual_std).clip(lower=0)
        future["Max_Occupancy"] = (future["Forecasted_Occupancy"] + residual_std).clip(upper=1)
    future["Selected_Model"] = selected_model
    future["Feature_Profile"] = feature_profile
    return (
        future[
            [
                "Date",
                "Forecasted_Occupancy",
                "Min_Occupancy",
                "Max_Occupancy",
                "Competitor_Rate",
                "Selected_Model",
                "Feature_Profile",
            ]
        ],
        resolved_schema,
    )


def _plot_best_model_forecast(history: pd.DataFrame, forecast: pd.DataFrame, best_model: str, output_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_history = history.tail(180)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(plot_history["Date"], plot_history["Occupancy_Rate"], label="Actual Occupancy", color="#1f77b4", linewidth=2)
    ax.plot(forecast["Date"], forecast["Forecasted_Occupancy"], label=f"Forecast ({best_model})", color="#d62728", linewidth=2.5)
    ax.fill_between(
        forecast["Date"],
        forecast["Min_Occupancy"],
        forecast["Max_Occupancy"],
        color="#d62728",
        alpha=0.15,
        label="Forecast Range",
    )
    ax.axvline(plot_history["Date"].max(), color="#444444", linestyle="--", linewidth=1, label="Forecast Start")
    ax.set_title("Actual Occupancy + Champion Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Occupancy Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_lag_metrics(lag_metrics: pd.DataFrame, plots_dir: str):
    if lag_metrics.empty:
        return
    lag_metrics = _with_feature_profile(lag_metrics)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["MAE_pp", "RMSE_pp", "Bias_pp", "Stability"]:
        if metric not in lag_metrics.columns:
            continue
        fig, ax = plt.subplots(figsize=(14, 7))
        for (feature_profile, model_name), model_df in lag_metrics.groupby(["Feature_Profile", "Model"]):
            model_df = model_df.sort_values("Lag")
            ax.plot(model_df["Lag"], model_df[metric], label=f"{model_name} [{feature_profile}]", linewidth=1.8)
        ax.set_title(f"{metric} by Forecast Lag")
        ax.set_xlabel("Forecast Lag")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"lag_metric_{metric.lower()}.png"), dpi=180)
        plt.close(fig)


def _plot_backtest_scenario(predictions: pd.DataFrame, plots_dir: str):
    if predictions.empty:
        return
    predictions = _with_feature_profile(predictions)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_cutoff = pd.to_datetime(predictions["Cutoff"]).max()
    latest = predictions[pd.to_datetime(predictions["Cutoff"]).eq(latest_cutoff)].copy()
    actual = latest[["Date", "Actual"]].drop_duplicates().sort_values("Date")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual["Date"], actual["Actual"], label="Actual", color="#111111", linewidth=3)
    for (feature_profile, model_name), model_df in latest.groupby(["Feature_Profile", "Model"]):
        model_df = model_df.sort_values("Date")
        ax.plot(model_df["Date"], model_df["Predicted"], label=f"{model_name} [{feature_profile}]", linewidth=1.6, alpha=0.85)
    ax.set_title(f"Actuals + Backtest Predictions (latest fold ending {latest_cutoff.date()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Occupancy Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "backtest_models_latest_scenario.png"), dpi=180)
    plt.close(fig)


def _plot_backtest_timeline(folds: pd.DataFrame, output_path: str):
    if folds.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_folds = folds.copy().reset_index(drop=True)
    fig_height = max(8, min(16, len(plot_folds) * 0.18))
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for idx, row in plot_folds.iterrows():
        y = len(plot_folds) - idx
        train_start = mdates.date2num(pd.to_datetime(row["Train_Start"]))
        train_end = mdates.date2num(pd.to_datetime(row["Train_End"]))
        val_start = mdates.date2num(pd.to_datetime(row["Validation_Start"]))
        val_end = mdates.date2num(pd.to_datetime(row["Validation_End"]))
        edge = "#7b3fc6" if row["Split"] == "audit" else "none"
        linewidth = 1.3 if row["Split"] == "audit" else 0
        ax.broken_barh([(train_start, train_end - train_start)], (y - 0.35, 0.7), facecolors="#cfead6")
        ax.broken_barh(
            [(val_start, val_end - val_start)],
            (y - 0.35, 0.7),
            facecolors="#ffe7a8",
            edgecolors=edge,
            linewidth=linewidth,
        )

    audit_rows = plot_folds[plot_folds["Split"].eq("audit")]
    if not audit_rows.empty:
        audit_start_idx = int(audit_rows.index.min())
        y_top = len(plot_folds) - audit_start_idx + 0.5
        ax.axhspan(0.5, y_top, color="#efe6fb", alpha=0.35, zorder=-1)
        ax.text(
            pd.to_datetime(plot_folds["Validation_End"]).max(),
            y_top - 0.35,
            "audit folds",
            color="#6b2fb3",
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Weekly Rolling-Origin Backtest for a 30-Day Forecast Horizon", fontsize=15, weight="bold")
    ax.set_xlabel("Calendar time")
    ax.set_ylabel("Weekly folds")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor="#cfead6", label="expanding training history"),
            Patch(facecolor="#ffe7a8", label="30-day forecast window"),
            Patch(facecolor="#efe6fb", label="reserved audit region"),
        ],
        loc="upper left",
    )
    ax.text(
        0.99,
        0.02,
        "MAE points first; RMSE breaks practical ties",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_forecast_plots(
    daily_df: pd.DataFrame,
    forecast: pd.DataFrame,
    champion_model: str,
    lag_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    plots_dir: str,
):
    os.makedirs(plots_dir, exist_ok=True)
    history = _actuals(daily_df)
    _plot_best_model_forecast(history, forecast, champion_model, os.path.join(plots_dir, "champion_actuals_forecast.png"))
    _plot_backtest_scenario(predictions, plots_dir)
    _plot_lag_metrics(lag_metrics, plots_dir)


def _hyperparameter_tuning_config() -> HyperparameterTuningConfig:
    return HyperparameterTuningConfig(
        n_trials=DEFAULT_HYPERPARAM_TRIALS,
        recent_folds=DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS,
        mae_tie_threshold_pp=DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP,
    )


def _set_tuned_model_params(payload: dict) -> dict[str, dict]:
    global TUNED_MODEL_PARAMS
    TUNED_MODEL_PARAMS = tuned_params_from_payload(payload)
    return TUNED_MODEL_PARAMS


def _load_and_apply_hyperparameter_tuning(path: Optional[str]) -> dict:
    payload = load_tuning_payload(path) if path else {}
    tuned_params = _set_tuned_model_params(payload)
    if path and payload:
        if tuned_params:
            print(
                f"Loaded ML hyperparameter tuning artifact from {path} "
                f"with {len(tuned_params)} tuned model(s).",
                flush=True,
            )
        else:
            print(
                f"Loaded ML hyperparameter tuning artifact from {path}, "
                f"but no tuned model params were available (status: {payload.get('_status', 'unknown')}).",
                flush=True,
            )
    return payload


def _run_and_save_hyperparameter_tuning(
    daily_df: pd.DataFrame,
    models: Iterable[str],
    horizon: int,
    paths: dict,
) -> tuple[dict, pd.DataFrame]:
    tuning_path = paths.get("hyperparameter_tuning")
    report_path = paths.get("hyperparameter_tuning_report")
    config = _hyperparameter_tuning_config()
    tuner = ForecastHyperparameterTuner(config)
    payload, report = tuner.tune(daily_df, models=models, horizon=horizon)
    if tuning_path:
        save_tuning_payload(payload, tuning_path)
    if report_path:
        _safe_to_csv(report, report_path, index=False)
    _set_tuned_model_params(payload)
    if payload.get("_status") == "skipped_optuna_unavailable":
        print("Optuna is unavailable; using default ML hyperparameters.", flush=True)
    elif payload.get("_status") == "skipped_no_recent_folds":
        print("No recent full-horizon folds available; using default ML hyperparameters.", flush=True)
    elif tuning_path:
        print(f"Saved ML hyperparameter tuning artifact to {tuning_path}", flush=True)
    return payload, report


def _champion_tuning_metadata(champion: ForecastChampion, payload: dict, tuning_path: Optional[str]) -> dict:
    model_payload = payload.get(champion.model, {}) if isinstance(payload, dict) else {}
    return {
        "artifact_path": tuning_path,
        "objective": model_payload.get("objective", payload.get("_metadata", {}).get("objective")),
        "n_trials": model_payload.get("n_trials", payload.get("_metadata", {}).get("n_trials")),
        "recent_folds_used": model_payload.get("recent_folds_used", payload.get("_metadata", {}).get("recent_folds")),
        "data_end_date": model_payload.get("data_end_date", payload.get("_metadata", {}).get("data_end_date")),
        "tuned_at": model_payload.get("tuned_at", payload.get("_tuned_at")),
        "best_params": model_payload.get("best_params", {}),
        "best_mae_pp": model_payload.get("best_mae_pp"),
        "best_rmse_pp": model_payload.get("best_rmse_pp"),
        "best_bias_pp": model_payload.get("best_bias_pp"),
        "best_abs_bias_pp": model_payload.get("best_abs_bias_pp"),
        "best_wape": model_payload.get("best_wape"),
        "tuning_mae_tie_threshold_pp": model_payload.get(
            "tuning_mae_tie_threshold_pp",
            payload.get("_metadata", {}).get("mae_tie_threshold_pp"),
        ),
        "status": model_payload.get("status", payload.get("_status")),
    }


def run_backtest_and_save(
    daily_df: pd.DataFrame,
    paths: dict,
    horizon: int = DEFAULT_HORIZON,
    scenario_lags: Optional[Iterable[int]] = None,
    models: Optional[Iterable[str]] = None,
):
    requested_models = list(models or DEFAULT_MODELS)
    tuning_payload, _ = _run_and_save_hyperparameter_tuning(
        daily_df,
        models=requested_models,
        horizon=horizon,
        paths=paths,
    )
    (
        overall,
        lag_metrics,
        scenario_metrics,
        predictions,
        fold_metrics,
        folds,
        boruta_selection_report,
        production_schemas,
        production_schema_metadata,
    ) = run_backtest_detailed(
        daily_df,
        models=requested_models,
        horizon=horizon,
        scenario_lags=scenario_lags,
        return_feature_artifacts=True,
    )
    selection_predictions = predictions[predictions["Split"].eq("selection")].copy()
    audit_predictions = predictions[predictions["Split"].eq("audit")].copy()
    audit_fold_metrics = fold_metrics[fold_metrics["Split"].eq("audit")].copy()
    audit_summary = _aggregate_fold_metrics(audit_fold_metrics, split="audit")
    audit_lag_metrics = _metrics_from_predictions(
        audit_predictions,
        ["Feature_Profile", "Model", "Strategy", "Lag"],
    )
    champion, acceptance_metadata = _select_champion_with_acceptance(overall, audit_summary, horizon=horizon)

    _, interval_quantiles = _calibrate_interval_quantiles(
        selection_predictions,
        champion.model,
        feature_profile=champion.feature_profile,
        interval_level=DEFAULT_INTERVAL_LEVEL,
    )
    audit_interval_coverage = _interval_coverage(
        audit_predictions,
        champion.model,
        champion.feature_profile,
        interval_quantiles,
    )
    audit_interval_coverage_overall = (
        float(
            audit_interval_coverage["Interval_Coverage"].mul(audit_interval_coverage["Observations"]).sum()
            / audit_interval_coverage["Observations"].sum()
        )
        if not audit_interval_coverage.empty
        else np.nan
    )

    selection_mean_mae_pp = float(champion.metrics.get("MAE_pp", np.nan))
    champion_audit_row = audit_summary[
        audit_summary["Model"].eq(champion.model)
        & audit_summary["Feature_Profile"].eq(champion.feature_profile)
    ]
    audit_mean_mae_pp = float(champion_audit_row["MAE_pp"].iloc[0]) if not champion_audit_row.empty else np.nan
    audit_drift_ratio, audit_status = _audit_status(selection_mean_mae_pp, audit_mean_mae_pp)
    if not audit_summary.empty:
        audit_summary["Is_Champion"] = (
            audit_summary["Model"].eq(champion.model)
            & audit_summary["Feature_Profile"].eq(champion.feature_profile)
        )
        audit_summary["Selection_Mean_Fold_MAE_pp"] = np.nan
        audit_summary["Audit_Drift_Ratio"] = np.nan
        audit_summary["Audit_Status"] = ""
        audit_summary["Interval_Coverage"] = np.nan
        champion_mask = audit_summary["Is_Champion"]
        audit_summary.loc[champion_mask, "Selection_Mean_Fold_MAE_pp"] = selection_mean_mae_pp
        audit_summary.loc[champion_mask, "Audit_Drift_Ratio"] = audit_drift_ratio
        audit_summary.loc[champion_mask, "Audit_Status"] = audit_status
        audit_summary.loc[champion_mask, "Interval_Coverage"] = audit_interval_coverage_overall

    forecast, feature_schema = forecast_demand(
        daily_df,
        selected_model=champion.model,
        horizon_days=horizon,
        interval_quantiles=interval_quantiles,
        feature_profile=champion.feature_profile,
        feature_schema=production_schemas.get(champion.strategy) if _uses_boruta_features(champion.model, champion.feature_profile) else None,
    )
    selected_historical_features, mandatory_features = _split_feature_schema(feature_schema)
    feature_manifest = _feature_manifest(
        _actuals(daily_df),
        horizon,
        champion.feature_profile,
        feature_schema,
    )
    champion = ForecastChampion(
        model=champion.model,
        strategy=champion.strategy,
        horizon=champion.horizon,
        selected_at=champion.selected_at,
        metrics=champion.metrics,
        feature_schema=feature_schema,
        feature_profile=champion.feature_profile,
        selection_objective=champion.selection_objective,
        mae_tie_threshold_pp=champion.mae_tie_threshold_pp,
        selected_historical_features=selected_historical_features,
        mandatory_features=mandatory_features,
        feature_selection_metadata={
            "selector": "boruta" if _uses_boruta_features(champion.model, champion.feature_profile) else "none",
            "selection_mode": "one_time_full_history" if _uses_boruta_features(champion.model, champion.feature_profile) else "none",
            "chain_anchor_horizons": CHAIN_BORUTA_ANCHORS if champion.strategy == "regressor_chain" else [],
            "chain_min_anchor_count": CHAIN_BORUTA_MIN_ANCHORS if champion.strategy == "regressor_chain" else None,
            "schema_metadata": production_schema_metadata.get(champion.strategy, {}),
            "acceptance": acceptance_metadata,
        },
        backtest_cadence_days=champion.backtest_cadence_days,
        interval_level=DEFAULT_INTERVAL_LEVEL,
        interval_quantiles=interval_quantiles,
        hyperparameter_tuning_metadata=_champion_tuning_metadata(
            champion,
            tuning_payload,
            paths.get("hyperparameter_tuning"),
        ),
        backtest_metadata={
            "fold_step_days": DEFAULT_BACKTEST_STEP_DAYS,
            "min_train_days": DEFAULT_MIN_TRAIN_DAYS,
            "total_folds": int(len(folds)),
            "selection_folds": int(folds["Split"].eq("selection").sum()) if not folds.empty else 0,
            "audit_folds": int(folds["Split"].eq("audit").sum()) if not folds.empty else 0,
            "audit_drift_threshold": DEFAULT_AUDIT_DRIFT_THRESHOLD,
            "audit_status": audit_status,
            "selection_objective": MODEL_SELECTION_OBJECTIVE,
            "mae_tie_threshold_pp": MODEL_SELECTION_MAE_TIE_THRESHOLD_PP,
            "selection_mean_fold_mae_pp": selection_mean_mae_pp,
            "audit_mean_fold_mae_pp": audit_mean_mae_pp,
            "audit_drift_ratio": audit_drift_ratio,
            "audit_interval_coverage": audit_interval_coverage_overall,
        },
    )

    os.makedirs(os.path.dirname(paths["forecast"]), exist_ok=True)
    _safe_to_csv(forecast, paths["forecast"], index=False)
    comparison_export = _export_model_metrics(overall)
    _safe_to_csv(comparison_export, paths["comparison"], index=False)
    _safe_to_csv(comparison_export.head(1), paths["metrics"], index=False)
    _safe_to_csv(lag_metrics, paths["lag_metrics"], index=False)
    _safe_to_csv(scenario_metrics, paths["scenario_metrics"], index=False)
    _safe_to_csv(predictions, paths["predictions"], index=False)
    _safe_to_csv(fold_metrics, paths["fold_metrics"], index=False)
    _safe_to_csv(audit_predictions, paths["audit_predictions"], index=False)
    _safe_to_csv(audit_fold_metrics, paths["audit_fold_metrics"], index=False)
    _safe_to_csv(_export_model_metrics(audit_summary, include_audit_columns=True), paths["audit_summary"], index=False)
    _safe_to_csv(audit_lag_metrics, paths["audit_lag_metrics"], index=False)
    _safe_to_csv(audit_interval_coverage, paths["audit_interval_coverage"], index=False)
    _safe_to_csv(feature_manifest, paths["feature_manifest"], index=False)
    _safe_to_csv(boruta_selection_report, paths["boruta_selection_report"], index=False)
    save_champion(champion, paths["champion"])
    save_forecast_plots(daily_df, forecast, champion.model, lag_metrics, predictions, paths["plots_dir"])
    _plot_backtest_timeline(folds, paths["timeline_plot"])
    return forecast, overall, champion


def run_forecast_and_save(daily_df: pd.DataFrame, paths: dict, horizon: int = DEFAULT_HORIZON):
    champion = load_champion(paths["champion"], default_horizon=horizon)
    tuning_payload = _load_and_apply_hyperparameter_tuning(paths.get("hyperparameter_tuning"))
    selected_schema = champion.feature_schema or None
    selection_report = pd.DataFrame()
    schema_metadata = {}
    if selected_schema is None and _uses_boruta_features(champion.model, champion.feature_profile):
        history = _actuals(daily_df)
        model_specs = [{"model": champion.model, "feature_profile": ENHANCED_PROFILE}]
        production_schemas, selection_report, production_schema_metadata = _select_production_feature_schemas(
            history,
            champion.horizon,
            model_specs,
        )
        selected_schema = production_schemas.get(champion.strategy)
        schema_metadata = production_schema_metadata.get(champion.strategy, {})
    forecast, feature_schema = forecast_demand(
        daily_df,
        selected_model=champion.model,
        horizon_days=champion.horizon,
        interval_quantiles=champion.interval_quantiles,
        feature_profile=champion.feature_profile,
        feature_schema=selected_schema,
    )
    if feature_schema and feature_schema != champion.feature_schema:
        selected_historical_features, mandatory_features = _split_feature_schema(feature_schema)
        champion = ForecastChampion(
            model=champion.model,
            strategy=champion.strategy,
            horizon=champion.horizon,
            selected_at=champion.selected_at,
            metrics=champion.metrics,
            feature_schema=feature_schema,
            feature_profile=champion.feature_profile,
            selected_historical_features=selected_historical_features,
            mandatory_features=mandatory_features,
            feature_selection_metadata={
                **champion.feature_selection_metadata,
                "selector": "boruta" if _uses_boruta_features(champion.model, champion.feature_profile) else "none",
                "selection_mode": "one_time_full_history" if _uses_boruta_features(champion.model, champion.feature_profile) else "none",
                "schema_metadata": schema_metadata,
            },
            backtest_cadence_days=champion.backtest_cadence_days,
            interval_level=champion.interval_level,
            interval_quantiles=champion.interval_quantiles,
            backtest_metadata=champion.backtest_metadata,
            hyperparameter_tuning_metadata=_champion_tuning_metadata(
                champion,
                tuning_payload,
                paths.get("hyperparameter_tuning"),
            ),
        )
        save_champion(champion, paths["champion"])
        if "feature_manifest" in paths:
            _safe_to_csv(
                _feature_manifest(_actuals(daily_df), champion.horizon, champion.feature_profile, feature_schema),
                paths["feature_manifest"],
                index=False,
            )
        if "boruta_selection_report" in paths and not selection_report.empty:
            _safe_to_csv(selection_report, paths["boruta_selection_report"], index=False)
    os.makedirs(os.path.dirname(paths["forecast"]), exist_ok=True)
    _safe_to_csv(forecast, paths["forecast"], index=False)
    _plot_best_model_forecast(_actuals(daily_df), forecast, champion.model, os.path.join(paths["plots_dir"], "champion_actuals_forecast.png"))
    return forecast, champion


# Backward-compatible wrappers used by older app/scripts.
def run_backtest(daily_df: pd.DataFrame, models: Optional[Iterable[str]] = None, horizons=(7, 14, 30), cutoffs="rolling"):
    horizon = max(horizons) if horizons else DEFAULT_HORIZON
    overall, _, _, _, _, _ = run_backtest_detailed(daily_df, models=models, horizon=horizon)
    return overall


def save_forecast_artifacts(daily_df: pd.DataFrame, forecast_path: str, metrics_path: str, comparison_path: str, plots_dir: Optional[str] = None):
    paths = {
        "forecast": forecast_path,
        "metrics": metrics_path,
        "comparison": comparison_path,
        "lag_metrics": os.path.join(os.path.dirname(comparison_path), "backtest_lag_metrics.csv"),
        "scenario_metrics": os.path.join(os.path.dirname(comparison_path), "backtest_scenario_metrics.csv"),
        "predictions": os.path.join(os.path.dirname(comparison_path), "backtest_predictions.csv"),
        "fold_metrics": os.path.join(os.path.dirname(comparison_path), "backtest_fold_metrics.csv"),
        "audit_predictions": os.path.join(os.path.dirname(comparison_path), "backtest_audit_predictions.csv"),
        "audit_fold_metrics": os.path.join(os.path.dirname(comparison_path), "backtest_audit_fold_metrics.csv"),
        "audit_summary": os.path.join(os.path.dirname(comparison_path), "backtest_audit_summary.csv"),
        "audit_lag_metrics": os.path.join(os.path.dirname(comparison_path), "backtest_audit_lag_metrics.csv"),
        "audit_interval_coverage": os.path.join(os.path.dirname(comparison_path), "backtest_audit_interval_coverage.csv"),
        "feature_manifest": os.path.join(os.path.dirname(comparison_path), "feature_manifest.csv"),
        "boruta_selection_report": os.path.join(os.path.dirname(comparison_path), "boruta_selection_report.csv"),
        "champion": os.path.join(os.path.dirname(comparison_path), "forecast_champion.json"),
        "plots_dir": plots_dir or os.path.join(os.path.dirname(comparison_path), "plots"),
        "timeline_plot": os.path.join(os.path.dirname(os.path.dirname(comparison_path)), "docs", "backtest_timeline_explainer.png"),
    }
    forecast, metrics, _ = run_backtest_and_save(daily_df, paths)
    return forecast, metrics
