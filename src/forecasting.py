import json
import os
import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ForecastChampion:
    model: str
    strategy: str
    horizon: int
    selected_at: str
    metrics: dict
    feature_schema: list[str]
    backtest_cadence_days: int = 7


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
    df["Competitor_Rate"] = df["Competitor_Rate"].ffill().bfill().fillna(120.0)
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
    competitor_rate = pd.to_numeric(df.get("Competitor_Rate", 120.0), errors="coerce").ffill().bfill().fillna(120.0)
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
            "competitor_rate_scaled": competitor_rate.astype(float) / 100.0,
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
    features["recent_competitor_rate"] = float(history["Competitor_Rate"].tail(14).mean())
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
        features[f"{prefix}_competitor_rate"] = float(row.get("Competitor_Rate", features["recent_competitor_rate"]))
    return features


def _build_chain_training(history: pd.DataFrame, horizon: int, min_history: int = 90):
    rows = []
    targets = []
    max_origin = len(history) - horizon - 1
    for origin_idx in range(min_history, max_origin + 1):
        origin_history = history.iloc[: origin_idx + 1].copy()
        future = history.iloc[origin_idx + 1 : origin_idx + 1 + horizon].copy()
        rows.append(_feature_vector(origin_history, future, horizon=horizon))
        targets.append(future["Occupancy_Rate"].to_numpy())
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
        rows.append(_feature_vector(origin_history, target_row, horizon=1))
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


def _scenario_cutoffs(df: pd.DataFrame, scenario_lags: Iterable[int], horizon: int) -> list[tuple[int, pd.Timestamp]]:
    max_date = df["Date"].max()
    min_date = df["Date"].min()
    cutoffs = []
    for lag in scenario_lags:
        cutoff = max_date - pd.Timedelta(days=int(lag))
        if cutoff <= min_date + pd.Timedelta(days=120):
            continue
        available = len(df[(df["Date"].gt(cutoff)) & (df["Date"].le(cutoff + pd.Timedelta(days=horizon)))])
        if available > 0:
            cutoffs.append((int(lag), cutoff))
    return cutoffs


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


def run_backtest_detailed(
    daily_df: pd.DataFrame,
    models: Optional[Iterable[str]] = None,
    horizon: int = DEFAULT_HORIZON,
    scenario_lags: Optional[Iterable[int]] = None,
):
    df = _actuals(daily_df)
    models = list(models or DEFAULT_MODELS)
    scenario_lags = list(scenario_lags or DEFAULT_SCENARIO_LAGS)
    scenarios = _scenario_cutoffs(df, scenario_lags, horizon)

    prediction_rows = []
    for scenario_lag, cutoff in scenarios:
        train = df[df["Date"].le(cutoff)].copy()
        actual_future = df[(df["Date"].gt(cutoff)) & (df["Date"].le(cutoff + pd.Timedelta(days=horizon)))].copy()
        if actual_future.empty:
            continue
        print(f"Backtest scenario: train through {cutoff.date()} ({scenario_lag} days back), validate {len(actual_future)} days", flush=True)
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
                        "Scenario_Lag": scenario_lag,
                        "Cutoff": cutoff,
                        "Lag": lag_idx + 1,
                        "Date": actual_row["Date"],
                        "Actual": float(actual_row["Occupancy_Rate"]),
                        "Predicted": float(preds[lag_idx]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    overall = _metrics_from_predictions(predictions, ["Model", "Strategy"])
    lag_metrics = _metrics_from_predictions(predictions, ["Model", "Strategy", "Lag"])
    scenario_metrics = _metrics_from_predictions(predictions, ["Model", "Strategy", "Scenario_Lag", "Cutoff"])
    if not overall.empty:
        overall["Abs_Bias"] = overall["Bias"].abs()
        overall["Complexity"] = overall["Model"].map(MODEL_COMPLEXITY).fillna(5)
        overall = overall.sort_values(["WAPE", "Abs_Bias", "RMSE", "Complexity"], ascending=[True, True, True, True])
    return overall, lag_metrics, scenario_metrics, predictions


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
            },
            f,
            indent=4,
            default=str,
        )


def load_champion(path: str, default_horizon: int = DEFAULT_HORIZON) -> ForecastChampion:
    if not os.path.exists(path):
        return ForecastChampion(
            model="seasonal_naive_7",
            strategy="statistical",
            horizon=default_horizon,
            selected_at=pd.Timestamp.now().isoformat(),
            metrics={},
            feature_schema=[],
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
    )


def forecast_demand(
    daily_df: pd.DataFrame,
    selected_model: str = "seasonal_naive_7",
    horizon_days: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, list[str]]:
    history = _actuals(daily_df)
    preds, feature_schema = predict_model(history, horizon_days, selected_model)
    future = _future_dates(history, horizon_days)
    future["Forecasted_Occupancy"] = np.clip(preds, 0, 1)
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

    latest_scenario = predictions["Scenario_Lag"].min()
    latest = predictions[predictions["Scenario_Lag"].eq(latest_scenario)].copy()
    actual = latest[["Date", "Actual"]].drop_duplicates().sort_values("Date")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual["Date"], actual["Actual"], label="Actual", color="#111111", linewidth=3)
    for model_name, model_df in latest.groupby("Model"):
        model_df = model_df.sort_values("Date")
        ax.plot(model_df["Date"], model_df["Predicted"], label=model_name, linewidth=1.6, alpha=0.85)
    ax.set_title(f"Actuals + Backtest Predictions ({int(latest_scenario)} days back)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Occupancy Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "backtest_models_latest_scenario.png"), dpi=180)
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
    overall, lag_metrics, scenario_metrics, predictions = run_backtest_detailed(
        daily_df,
        models=models,
        horizon=horizon,
        scenario_lags=scenario_lags,
    )
    champion = select_champion(overall, horizon=horizon)
    forecast, feature_schema = forecast_demand(daily_df, selected_model=champion.model, horizon_days=horizon)
    champion = ForecastChampion(
        model=champion.model,
        strategy=champion.strategy,
        horizon=champion.horizon,
        selected_at=champion.selected_at,
        metrics=champion.metrics,
        feature_schema=feature_schema,
        backtest_cadence_days=champion.backtest_cadence_days,
    )

    os.makedirs(os.path.dirname(paths["forecast"]), exist_ok=True)
    forecast.to_csv(paths["forecast"], index=False)
    overall.to_csv(paths["comparison"], index=False)
    overall.head(1).to_csv(paths["metrics"], index=False)
    lag_metrics.to_csv(paths["lag_metrics"], index=False)
    scenario_metrics.to_csv(paths["scenario_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    scenario_metrics.to_csv(paths["fold_metrics"], index=False)
    save_champion(champion, paths["champion"])
    save_forecast_plots(daily_df, forecast, champion.model, lag_metrics, predictions, paths["plots_dir"])
    return forecast, overall, champion


def run_forecast_and_save(daily_df: pd.DataFrame, paths: dict, horizon: int = DEFAULT_HORIZON):
    champion = load_champion(paths["champion"], default_horizon=horizon)
    forecast, feature_schema = forecast_demand(daily_df, selected_model=champion.model, horizon_days=champion.horizon)
    if feature_schema and not champion.feature_schema:
        champion = ForecastChampion(
            model=champion.model,
            strategy=champion.strategy,
            horizon=champion.horizon,
            selected_at=champion.selected_at,
            metrics=champion.metrics,
            feature_schema=feature_schema,
            backtest_cadence_days=champion.backtest_cadence_days,
        )
        save_champion(champion, paths["champion"])
    os.makedirs(os.path.dirname(paths["forecast"]), exist_ok=True)
    forecast.to_csv(paths["forecast"], index=False)
    _plot_best_model_forecast(_actuals(daily_df), forecast, champion.model, os.path.join(paths["plots_dir"], "champion_actuals_forecast.png"))
    return forecast, champion


# Backward-compatible wrappers used by older app/scripts.
def run_backtest(daily_df: pd.DataFrame, models: Optional[Iterable[str]] = None, horizons=(7, 14, 30), cutoffs="rolling"):
    horizon = max(horizons) if horizons else DEFAULT_HORIZON
    overall, _, _, _ = run_backtest_detailed(daily_df, models=models, horizon=horizon)
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
        "champion": os.path.join(os.path.dirname(comparison_path), "forecast_champion.json"),
        "plots_dir": plots_dir or os.path.join(os.path.dirname(comparison_path), "plots"),
    }
    forecast, metrics, _ = run_backtest_and_save(daily_df, paths)
    return forecast, metrics
