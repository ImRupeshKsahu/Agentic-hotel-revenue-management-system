import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

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
    backtest_cadence_days: int = 7
    interval_level: float = DEFAULT_INTERVAL_LEVEL
    interval_quantiles: dict = field(default_factory=dict)
    backtest_metadata: dict = field(default_factory=dict)


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
    bias = np.mean(error)

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "WAPE": float(wape),
        "Bias": float(bias),
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
    dow = date.dayofweek
    doy = date.dayofyear
    month = date.month
    return {
        f"{prefix}_dow_sin": np.sin(2 * np.pi * dow / 7),
        f"{prefix}_dow_cos": np.cos(2 * np.pi * dow / 7),
        f"{prefix}_doy_sin": np.sin(2 * np.pi * doy / 365.25),
        f"{prefix}_doy_cos": np.cos(2 * np.pi * doy / 365.25),
        f"{prefix}_month_sin": np.sin(2 * np.pi * month / 12),
        f"{prefix}_month_cos": np.cos(2 * np.pi * month / 12),
        f"{prefix}_is_weekend": int(dow in [5, 6]),
    }


def _feature_vector(history: pd.DataFrame, future: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    y = history["Occupancy_Rate"].astype(float)
    features = {}
    for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
        features[f"lag_{lag}"] = float(y.iloc[-lag]) if len(y) >= lag else float(y.mean())
    for window in [7, 14, 28, 56]:
        tail = y.tail(window)
        features[f"roll_mean_{window}"] = float(tail.mean())
        features[f"roll_std_{window}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0
    features["trend_7"] = features["roll_mean_7"] - features["roll_mean_28"]
    features["trend_14"] = features["roll_mean_14"] - features["roll_mean_56"]
    features["recent_min_28"] = float(y.tail(28).min())
    features["recent_max_28"] = float(y.tail(28).max())
    features["recent_booking_pace"] = float(history["Booking_Pace"].tail(14).mean())
    features["recent_cancellations"] = float(history["Cancellations"].tail(14).mean())
    features.update(_calendar_features(pd.to_datetime(history["Date"].iloc[-1]), "origin"))

    padded_future = future.copy()
    if len(padded_future) < horizon:
        last_date = pd.to_datetime(history["Date"].iloc[-1])
        padded_future = _future_dates(history, horizon)

    for i in range(1, horizon + 1):
        row = padded_future.iloc[i - 1]
        date = pd.to_datetime(row["Date"])
        prefix = f"h{i}"
        features.update(_calendar_features(date, prefix))
        features[f"{prefix}_local_event"] = float(row.get("Local_Event", 0))
    return features


def _build_chain_training(history: pd.DataFrame, horizon: int, min_history: int = 90):
    rows = []
    targets = []
    max_origin = len(history) - horizon - 1
    for origin_idx in range(min_history, max_origin + 1):
        origin_history = history.iloc[: origin_idx + 1].copy()
        realized_future = history.iloc[origin_idx + 1 : origin_idx + 1 + horizon].copy()
        planned_future = _future_dates(origin_history, horizon)
        rows.append(_feature_vector(origin_history, planned_future, horizon=horizon))
        targets.append(realized_future["Occupancy_Rate"].to_numpy())
    if not rows:
        return pd.DataFrame(), np.empty((0, horizon)), []
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    return x, np.vstack(targets), list(x.columns)


def _build_recursive_training(history: pd.DataFrame, min_history: int = 90):
    rows = []
    targets = []
    for target_idx in range(min_history + 1, len(history)):
        origin_history = history.iloc[:target_idx].copy()
        target_row = history.iloc[[target_idx]].copy()
        planned_future = _future_dates(origin_history, 1)
        rows.append(_feature_vector(origin_history, planned_future, horizon=1))
        targets.append(float(target_row["Occupancy_Rate"].iloc[0]))
    if not rows:
        return pd.DataFrame(), np.array([]), []
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    return x, np.array(targets), list(x.columns)


def _base_estimator(model_name: str):
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
        return RandomForestRegressor(n_estimators=18, min_samples_leaf=6, random_state=42, n_jobs=-1)
    if model_name.startswith("extra_trees"):
        if ExtraTreesRegressor is None:
            return None
        return ExtraTreesRegressor(n_estimators=24, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if model_name.startswith("xgboost"):
        if XGBRegressor is None:
            return None
        return XGBRegressor(
            n_estimators=12,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
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


def _predict_chain(history: pd.DataFrame, horizon: int, model_name: str):
    if RegressorChain is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    estimator = _base_estimator(model_name)
    if estimator is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    x_train, y_train, feature_schema = _build_chain_training(history, horizon=horizon)
    if len(x_train) < 30:
        return _predict_statistical(history, horizon, "rolling_mean_7"), feature_schema
    model = RegressorChain(estimator, order=list(range(horizon)))
    model.fit(x_train, y_train)
    future = _future_dates(history, horizon)
    x_future = pd.DataFrame([_feature_vector(history, future, horizon=horizon)]).reindex(columns=feature_schema).fillna(0)
    return np.asarray(model.predict(x_future)[0], dtype=float), feature_schema


def _predict_recursive(history: pd.DataFrame, horizon: int, model_name: str):
    estimator = _base_estimator(model_name)
    if estimator is None:
        return _predict_statistical(history, horizon, "rolling_mean_7"), []
    x_train, y_train, feature_schema = _build_recursive_training(history)
    if len(x_train) < 30:
        return _predict_statistical(history, horizon, "rolling_mean_7"), feature_schema
    estimator.fit(x_train, y_train)

    recursive = history.copy()
    preds = []
    for step in range(horizon):
        future = _future_dates(recursive, 1)
        x_future = pd.DataFrame([_feature_vector(recursive, future, horizon=1)]).reindex(columns=feature_schema).fillna(0)
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


def predict_model(history: pd.DataFrame, horizon: int, model_name: str):
    if model_name in STATISTICAL_MODELS:
        return np.clip(_predict_statistical(history, horizon, model_name), 0, 1), []
    if model_name.endswith("_chain"):
        preds, schema = _predict_chain(history, horizon, model_name)
        return np.clip(preds, 0, 1), schema
    if model_name.endswith("_recursive"):
        preds, schema = _predict_recursive(history, horizon, model_name)
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


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = fold_metrics[fold_metrics["Split"].eq(split)].copy()
    if subset.empty:
        return pd.DataFrame()
    metric_cols = ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "Bias", "Accuracy", "Volatility", "Stability"]
    grouped = (
        subset.groupby(["Model", "Strategy"], as_index=False)
        .agg(
            Folds=("Fold_ID", "nunique"),
            Observations=("Observations", "sum"),
            **{col: (col, "mean") for col in metric_cols},
        )
    )
    return grouped


def _calibrate_interval_quantiles(
    predictions: pd.DataFrame,
    model_name: str,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> tuple[pd.DataFrame, dict]:
    model_predictions = predictions[predictions["Model"].eq(model_name)].copy()
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


def _interval_coverage(predictions: pd.DataFrame, model_name: str, interval_quantiles: dict) -> pd.DataFrame:
    model_predictions = predictions[predictions["Model"].eq(model_name)].copy()
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
    selection_mean_wape: float,
    audit_mean_wape: float,
    drift_threshold: float = DEFAULT_AUDIT_DRIFT_THRESHOLD,
) -> tuple[float, str]:
    drift_ratio = (
        float(audit_mean_wape / selection_mean_wape)
        if np.isfinite(selection_mean_wape) and selection_mean_wape > 0 and np.isfinite(audit_mean_wape)
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
        for model_name in models:
            print(f"  fitting {model_name}", flush=True)
            preds, _ = predict_model(train, horizon, model_name)
            valid_len = min(len(actual_future), len(preds))
            for lag_idx in range(valid_len):
                actual_row = actual_future.iloc[lag_idx]
                prediction_rows.append(
                    {
                        "Model": model_name,
                        "Strategy": _strategy_for_model(model_name),
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
    fold_metrics = _metrics_from_predictions(predictions, ["Split", "Fold_ID", "Model", "Strategy", "Cutoff"])
    overall = _aggregate_fold_metrics(fold_metrics, split="selection")
    lag_metrics = _metrics_from_predictions(predictions[predictions["Split"].eq("selection")], ["Model", "Strategy", "Lag"])
    scenario_metrics = fold_metrics.copy()
    if not overall.empty:
        overall["Abs_Bias"] = overall["Bias"].abs()
        overall["Complexity"] = overall["Model"].map(MODEL_COMPLEXITY).fillna(5)
        overall = overall.sort_values(["WAPE", "Abs_Bias", "RMSE", "Complexity"], ascending=[True, True, True, True])
    return overall, lag_metrics, scenario_metrics, predictions, fold_metrics, folds


def select_champion(overall_metrics: pd.DataFrame, horizon: int, feature_schema: list[str] | None = None) -> ForecastChampion:
    if overall_metrics.empty:
        model_name = "seasonal_naive_7"
        metrics = {}
    else:
        winner = overall_metrics.sort_values(["WAPE", "Abs_Bias", "RMSE", "Complexity"]).iloc[0]
        model_name = winner["Model"]
        metrics = winner.drop(labels=[c for c in ["Abs_Bias", "Complexity"] if c in winner.index]).to_dict()
    return ForecastChampion(
        model=model_name,
        strategy=_strategy_for_model(model_name),
        horizon=horizon,
        selected_at=pd.Timestamp.now().isoformat(),
        metrics=metrics,
        feature_schema=feature_schema or [],
    )


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
                "backtest_cadence_days": champion.backtest_cadence_days,
                "interval_level": champion.interval_level,
                "interval_quantiles": champion.interval_quantiles,
                "backtest_metadata": champion.backtest_metadata,
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
            interval_level=DEFAULT_INTERVAL_LEVEL,
            interval_quantiles={},
            backtest_metadata={},
        )
    with open(path, "r") as f:
        data = json.load(f)
    return ForecastChampion(
        model=data.get("model", "seasonal_naive_7"),
        strategy=data.get("strategy", _strategy_for_model(data.get("model", "seasonal_naive_7"))),
        horizon=int(data.get("horizon", default_horizon)),
        selected_at=data.get("selected_at", pd.Timestamp.now().isoformat()),
        metrics=data.get("metrics", {}),
        feature_schema=data.get("feature_schema", []),
        backtest_cadence_days=int(data.get("backtest_cadence_days", 7)),
        interval_level=float(data.get("interval_level", DEFAULT_INTERVAL_LEVEL)),
        interval_quantiles=data.get("interval_quantiles", {}),
        backtest_metadata=data.get("backtest_metadata", {}),
    )


def forecast_demand(
    daily_df: pd.DataFrame,
    selected_model: str = "seasonal_naive_7",
    horizon_days: int = DEFAULT_HORIZON,
    interval_quantiles: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str]]:
    history = _actuals(daily_df)
    preds, feature_schema = predict_model(history, horizon_days, selected_model)
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
    return future[["Date", "Forecasted_Occupancy", "Min_Occupancy", "Max_Occupancy", "Competitor_Rate", "Selected_Model"]], feature_schema


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["WAPE", "Bias", "Stability"]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for model_name, model_df in lag_metrics.groupby("Model"):
            model_df = model_df.sort_values("Lag")
            ax.plot(model_df["Lag"], model_df[metric], label=model_name, linewidth=1.8)
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_cutoff = pd.to_datetime(predictions["Cutoff"]).max()
    latest = predictions[pd.to_datetime(predictions["Cutoff"]).eq(latest_cutoff)].copy()
    actual = latest[["Date", "Actual"]].drop_duplicates().sort_values("Date")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual["Date"], actual["Actual"], label="Actual", color="#111111", linewidth=3)
    for model_name, model_df in latest.groupby("Model"):
        model_df = model_df.sort_values("Date")
        ax.plot(model_df["Date"], model_df["Predicted"], label=model_name, linewidth=1.6, alpha=0.85)
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
        "One WAPE per fold → mean fold WAPE by model",
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


def run_backtest_and_save(
    daily_df: pd.DataFrame,
    paths: dict,
    horizon: int = DEFAULT_HORIZON,
    scenario_lags: Optional[Iterable[int]] = None,
    models: Optional[Iterable[str]] = None,
):
    overall, lag_metrics, scenario_metrics, predictions, fold_metrics, folds = run_backtest_detailed(
        daily_df,
        models=models,
        horizon=horizon,
        scenario_lags=scenario_lags,
    )
    champion = select_champion(overall, horizon=horizon)
    selection_predictions = predictions[predictions["Split"].eq("selection")].copy()
    audit_predictions = predictions[predictions["Split"].eq("audit")].copy()
    audit_fold_metrics = fold_metrics[fold_metrics["Split"].eq("audit")].copy()
    audit_summary = _aggregate_fold_metrics(audit_fold_metrics, split="audit")
    audit_lag_metrics = _metrics_from_predictions(audit_predictions, ["Model", "Strategy", "Lag"])

    _, interval_quantiles = _calibrate_interval_quantiles(
        selection_predictions,
        champion.model,
        interval_level=DEFAULT_INTERVAL_LEVEL,
    )
    audit_interval_coverage = _interval_coverage(audit_predictions, champion.model, interval_quantiles)
    audit_interval_coverage_overall = (
        float(
            audit_interval_coverage["Interval_Coverage"].mul(audit_interval_coverage["Observations"]).sum()
            / audit_interval_coverage["Observations"].sum()
        )
        if not audit_interval_coverage.empty
        else np.nan
    )

    selection_mean_wape = float(champion.metrics.get("WAPE", np.nan))
    champion_audit_row = audit_summary[audit_summary["Model"].eq(champion.model)]
    audit_mean_wape = float(champion_audit_row["WAPE"].iloc[0]) if not champion_audit_row.empty else np.nan
    audit_drift_ratio, audit_status = _audit_status(selection_mean_wape, audit_mean_wape)
    if not audit_summary.empty:
        audit_summary["Is_Champion"] = audit_summary["Model"].eq(champion.model)
        audit_summary["Selection_Mean_Fold_WAPE"] = np.nan
        audit_summary["Audit_Drift_Ratio"] = np.nan
        audit_summary["Audit_Status"] = ""
        audit_summary["Interval_Coverage"] = np.nan
        champion_mask = audit_summary["Is_Champion"]
        audit_summary.loc[champion_mask, "Selection_Mean_Fold_WAPE"] = selection_mean_wape
        audit_summary.loc[champion_mask, "Audit_Drift_Ratio"] = audit_drift_ratio
        audit_summary.loc[champion_mask, "Audit_Status"] = audit_status
        audit_summary.loc[champion_mask, "Interval_Coverage"] = audit_interval_coverage_overall

    forecast, feature_schema = forecast_demand(
        daily_df,
        selected_model=champion.model,
        horizon_days=horizon,
        interval_quantiles=interval_quantiles,
    )
    champion = ForecastChampion(
        model=champion.model,
        strategy=champion.strategy,
        horizon=champion.horizon,
        selected_at=champion.selected_at,
        metrics=champion.metrics,
        feature_schema=feature_schema,
        backtest_cadence_days=champion.backtest_cadence_days,
        interval_level=DEFAULT_INTERVAL_LEVEL,
        interval_quantiles=interval_quantiles,
        backtest_metadata={
            "fold_step_days": DEFAULT_BACKTEST_STEP_DAYS,
            "min_train_days": DEFAULT_MIN_TRAIN_DAYS,
            "total_folds": int(len(folds)),
            "selection_folds": int(folds["Split"].eq("selection").sum()) if not folds.empty else 0,
            "audit_folds": int(folds["Split"].eq("audit").sum()) if not folds.empty else 0,
            "audit_drift_threshold": DEFAULT_AUDIT_DRIFT_THRESHOLD,
            "audit_status": audit_status,
            "selection_mean_fold_wape": selection_mean_wape,
            "audit_mean_fold_wape": audit_mean_wape,
            "audit_drift_ratio": audit_drift_ratio,
            "audit_interval_coverage": audit_interval_coverage_overall,
        },
    )

    os.makedirs(os.path.dirname(paths["forecast"]), exist_ok=True)
    _safe_to_csv(forecast, paths["forecast"], index=False)
    _safe_to_csv(overall, paths["comparison"], index=False)
    _safe_to_csv(overall.head(1), paths["metrics"], index=False)
    _safe_to_csv(lag_metrics, paths["lag_metrics"], index=False)
    _safe_to_csv(scenario_metrics, paths["scenario_metrics"], index=False)
    _safe_to_csv(predictions, paths["predictions"], index=False)
    _safe_to_csv(fold_metrics, paths["fold_metrics"], index=False)
    _safe_to_csv(audit_predictions, paths["audit_predictions"], index=False)
    _safe_to_csv(audit_fold_metrics, paths["audit_fold_metrics"], index=False)
    _safe_to_csv(audit_summary, paths["audit_summary"], index=False)
    _safe_to_csv(audit_lag_metrics, paths["audit_lag_metrics"], index=False)
    _safe_to_csv(audit_interval_coverage, paths["audit_interval_coverage"], index=False)
    save_champion(champion, paths["champion"])
    save_forecast_plots(daily_df, forecast, champion.model, lag_metrics, predictions, paths["plots_dir"])
    _plot_backtest_timeline(folds, paths["timeline_plot"])
    return forecast, overall, champion


def run_forecast_and_save(daily_df: pd.DataFrame, paths: dict, horizon: int = DEFAULT_HORIZON):
    champion = load_champion(paths["champion"], default_horizon=horizon)
    forecast, feature_schema = forecast_demand(
        daily_df,
        selected_model=champion.model,
        horizon_days=champion.horizon,
        interval_quantiles=champion.interval_quantiles,
    )
    if feature_schema and feature_schema != champion.feature_schema:
        champion = ForecastChampion(
            model=champion.model,
            strategy=champion.strategy,
            horizon=champion.horizon,
            selected_at=champion.selected_at,
            metrics=champion.metrics,
            feature_schema=feature_schema,
            backtest_cadence_days=champion.backtest_cadence_days,
            interval_level=champion.interval_level,
            interval_quantiles=champion.interval_quantiles,
            backtest_metadata=champion.backtest_metadata,
        )
        save_champion(champion, paths["champion"])
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
        "champion": os.path.join(os.path.dirname(comparison_path), "forecast_champion.json"),
        "plots_dir": plots_dir or os.path.join(os.path.dirname(comparison_path), "plots"),
        "timeline_plot": os.path.join(os.path.dirname(os.path.dirname(comparison_path)), "docs", "backtest_timeline_explainer.png"),
    }
    forecast, metrics, _ = run_backtest_and_save(daily_df, paths)
    return forecast, metrics
