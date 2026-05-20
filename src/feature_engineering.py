from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from config import (
    FORECAST_BASELINE_LAGS,
    FORECAST_BASELINE_ROLLING_STATS,
    FORECAST_BASELINE_TREND_FEATURES,
    FORECAST_ENHANCED_EXTRA_LAGS,
    FORECAST_ENHANCED_ROLLING_STATS,
    FORECAST_FORCE_KEEP_FEATURES,
    FORECAST_FUTURE_KNOWN_FEATURES,
    FORECAST_OPERATIONAL_SIGNAL_COLUMNS,
    FORECAST_OPERATIONAL_STATS,
    FORECAST_OPERATIONAL_WINDOWS,
    FORECAST_ORIGIN_CALENDAR_FEATURES,
    FORECAST_RECENT_EXTREME_WINDOW,
    FORECAST_RECENT_OPERATIONAL_WINDOW,
    FORECAST_ROLLING_WINDOWS,
    FORECAST_SEASONAL_INDEX_FEATURES,
    FORECAST_TREND_PROJECTION_FEATURES,
)


class FeatureEngineer:
    """Build model features from the feature contract defined in config.py."""

    BASELINE_PROFILE = "baseline"
    ENHANCED_PROFILE = "enhanced_v1"
    BORUTA_SELECTED_PROFILE = "boruta_selected"
    STATISTICAL_PROFILE = "statistical"
    BORUTA_FEATURE_PROFILES = {ENHANCED_PROFILE, BORUTA_SELECTED_PROFILE}

    def __init__(self, future_factory: Callable[[pd.DataFrame, int], pd.DataFrame] | None = None):
        self.future_factory = future_factory

    @property
    def force_keep_features(self) -> list[str]:
        return list(FORECAST_FORCE_KEEP_FEATURES)

    def mandatory_feature_names(self, columns: Iterable[str]) -> list[str]:
        return [col for col in columns if self._is_future_known_feature(col)]

    def split_feature_schema(self, feature_schema: Iterable[str]) -> tuple[list[str], list[str]]:
        schema = list(feature_schema)
        mandatory = self.mandatory_feature_names(schema)
        selected_historical = [feature for feature in schema if feature not in mandatory]
        return selected_historical, mandatory

    def feature_vector(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        horizon: int,
        feature_profile: str = BASELINE_PROFILE,
    ) -> dict:
        features = self._baseline_feature_vector(history, future, horizon=horizon)
        if feature_profile in self.BORUTA_FEATURE_PROFILES:
            features.update(self._enhanced_only_features(history, features))
        return features

    def _baseline_feature_vector(self, history: pd.DataFrame, future: pd.DataFrame, horizon: int) -> dict:
        y = history["Occupancy_Rate"].astype(float)
        features = {}

        for lag in FORECAST_BASELINE_LAGS:
            features[f"lag_{lag}"] = self._series_value_or_mean(y, lag)

        self._add_rolling_features(features, y, FORECAST_ROLLING_WINDOWS, FORECAST_BASELINE_ROLLING_STATS)

        for feature_name, (left, right) in FORECAST_BASELINE_TREND_FEATURES.items():
            features[feature_name] = features[left] - features[right]

        extreme_window = FORECAST_RECENT_EXTREME_WINDOW
        features[f"recent_min_{extreme_window}"] = float(y.tail(extreme_window).min())
        features[f"recent_max_{extreme_window}"] = float(y.tail(extreme_window).max())
        features["recent_booking_pace"] = float(history["Booking_Pace"].tail(FORECAST_RECENT_OPERATIONAL_WINDOW).mean())
        features["recent_cancellations"] = float(history["Cancellations"].tail(FORECAST_RECENT_OPERATIONAL_WINDOW).mean())

        features.update(self.calendar_features(pd.to_datetime(history["Date"].iloc[-1]), "origin", FORECAST_ORIGIN_CALENDAR_FEATURES))
        for i, row in enumerate(self._padded_future(history, future, horizon).itertuples(index=False), start=1):
            date = pd.to_datetime(row.Date)
            prefix = f"h{i}"
            features.update(self.calendar_features(date, prefix, FORECAST_FUTURE_KNOWN_FEATURES))
            features[f"{prefix}_local_event"] = float(getattr(row, "Local_Event", 0))
        return features

    def _enhanced_only_features(self, history: pd.DataFrame, baseline_features: dict) -> dict:
        y = history["Occupancy_Rate"].astype(float)
        features = {}

        for lag in FORECAST_ENHANCED_EXTRA_LAGS:
            features[f"lag_{lag}"] = self._series_value_or_mean(y, lag)

        self._add_rolling_features(features, y, FORECAST_ROLLING_WINDOWS, FORECAST_ENHANCED_ROLLING_STATS)

        for feature_name, (base_feature, slope_feature, horizon_days) in FORECAST_TREND_PROJECTION_FEATURES.items():
            features[feature_name] = baseline_features.get(base_feature, 0.0) + features.get(slope_feature, 0.0) * horizon_days

        features["wow_mean_diff_7"] = float(y.tail(7).mean() - y.iloc[-14:-7].mean()) if len(y) >= 14 else 0.0
        features["wow_level_diff_7"] = self._series_value_or_mean(y, 1) - self._series_value_or_mean(y, 8)
        features["yoy_level_diff_364"] = self._series_value_or_mean(y, 1) - self._series_value_or_mean(y, 364)
        features["yoy_mean_diff_28"] = (
            float(y.tail(28).mean() - y.iloc[-392:-364].mean())
            if len(y) >= 392
            else 0.0
        )

        origin_date = pd.to_datetime(history["Date"].iloc[-1])
        if "dow_seasonal_index" in FORECAST_SEASONAL_INDEX_FEATURES:
            features["dow_seasonal_index"] = self._seasonal_index(history, "dow", origin_date.dayofweek)
        if "month_seasonal_index" in FORECAST_SEASONAL_INDEX_FEATURES:
            features["month_seasonal_index"] = self._seasonal_index(history, "month", origin_date.month)

        for signal_name, column in FORECAST_OPERATIONAL_SIGNAL_COLUMNS.items():
            series = history[column].astype(float)
            for window in FORECAST_OPERATIONAL_WINDOWS:
                tail = series.tail(window)
                for stat in FORECAST_OPERATIONAL_STATS:
                    features[f"{signal_name}_{stat}_{window}"] = self._series_stat(tail, stat)
        return features

    def _add_rolling_features(self, features: dict, series: pd.Series, windows: Iterable[int], stats: Iterable[str]) -> None:
        for window in windows:
            tail = series.tail(window)
            for stat in stats:
                features[f"roll_{stat}_{window}"] = self._series_stat(tail, stat, full_series=series, window=window)

    def _padded_future(self, history: pd.DataFrame, future: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if len(future) >= horizon:
            return future.copy().iloc[:horizon]
        if self.future_factory is None:
            raise ValueError("future_factory is required when future rows are shorter than horizon")
        return self.future_factory(history, horizon)

    def _is_future_known_feature(self, feature_name: str) -> bool:
        return feature_name.startswith("h") and "_" in feature_name

    def calendar_features(self, date: pd.Timestamp, prefix: str, feature_names: Iterable[str] | None = None) -> dict:
        feature_names = list(feature_names or FORECAST_ORIGIN_CALENDAR_FEATURES)
        dow = date.dayofweek
        doy = date.dayofyear
        month = date.month
        values = {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_weekend": int(dow in [5, 6]),
        }
        return {f"{prefix}_{name}": values[name] for name in feature_names if name in values}

    def _series_stat(self, series: pd.Series, stat: str, full_series: pd.Series | None = None, window: int | None = None) -> float:
        if stat == "mean":
            return float(series.mean())
        if stat == "std":
            return float(series.std(ddof=0)) if len(series) > 1 else 0.0
        if stat == "min":
            return float(series.min())
        if stat == "max":
            return float(series.max())
        if stat == "sum":
            return float(series.sum())
        if stat == "slope":
            if full_series is None or window is None:
                return self._rolling_slope(series)
            return self._rolling_slope(full_series.tail(window))
        raise ValueError(f"Unsupported feature stat: {stat}")

    def _series_value_or_mean(self, series: pd.Series, lag: int) -> float:
        if len(series) >= lag:
            return float(series.iloc[-lag])
        mean_value = float(series.mean()) if len(series) else 0.0
        return 0.0 if pd.isna(mean_value) else mean_value

    def _rolling_slope(self, series: pd.Series) -> float:
        tail = series.astype(float)
        if len(tail) < 2:
            return 0.0
        x = np.arange(len(tail), dtype=float)
        return float(np.polyfit(x, tail.to_numpy(dtype=float), deg=1)[0])

    def _seasonal_index(self, history: pd.DataFrame, group: str, value) -> float:
        overall = float(history["Occupancy_Rate"].mean())
        if not np.isfinite(overall) or abs(overall) < 1e-9:
            return 1.0
        dates = pd.to_datetime(history["Date"])
        if group == "dow":
            grouped = history.assign(_group=dates.dt.dayofweek).groupby("_group")["Occupancy_Rate"].mean()
        elif group == "month":
            grouped = history.assign(_group=dates.dt.month).groupby("_group")["Occupancy_Rate"].mean()
        else:
            return 1.0
        seasonal_value = float(grouped.get(value, overall))
        return seasonal_value / overall if np.isfinite(seasonal_value) else 1.0
