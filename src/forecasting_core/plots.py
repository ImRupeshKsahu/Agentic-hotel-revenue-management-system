from __future__ import annotations

import pandas as pd

import forecasting_core.legacy as legacy


class ForecastPlotter:
    """Plot creation for forecast and backtest artifacts."""

    def best_model_forecast(self, history: pd.DataFrame, forecast: pd.DataFrame, best_model: str, output_path: str) -> None:
        legacy._plot_best_model_forecast(history, forecast, best_model, output_path)

    def lag_metrics(self, lag_metrics: pd.DataFrame, plots_dir: str) -> None:
        legacy._plot_lag_metrics(lag_metrics, plots_dir)

    def backtest_scenario(self, predictions: pd.DataFrame, plots_dir: str) -> None:
        legacy._plot_backtest_scenario(predictions, plots_dir)

    def backtest_timeline(self, folds: pd.DataFrame, output_path: str) -> None:
        legacy._plot_backtest_timeline(folds, output_path)

    def save_forecast_plots(
        self,
        daily_df: pd.DataFrame,
        forecast: pd.DataFrame,
        champion_model: str,
        lag_metrics: pd.DataFrame,
        predictions: pd.DataFrame,
        plots_dir: str,
    ) -> None:
        legacy.save_forecast_plots(daily_df, forecast, champion_model, lag_metrics, predictions, plots_dir)
