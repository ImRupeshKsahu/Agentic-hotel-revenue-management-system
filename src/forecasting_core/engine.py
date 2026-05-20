from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import pandas as pd

from feature_engineering import FeatureEngineer
from forecasting_core.algorithms import algorithm_for_model
from forecasting_core.artifacts import ArtifactStore, ForecastChampion
from forecasting_core.backtesting import ForecastBacktester
from forecasting_core.boruta_selector import BorutaFeatureSelector
from forecasting_core.config import ForecastRunConfig
import forecasting_core.legacy as legacy
from forecasting_core.model_registry import ForecastModelRegistry
from forecasting_core.plots import ForecastPlotter


class ForecastEngine:
    """High-level orchestration for forecasting, model competition, and artifacts."""

    def __init__(
        self,
        config: ForecastRunConfig | None = None,
        feature_engineer: FeatureEngineer | None = None,
        model_registry: ForecastModelRegistry | None = None,
        feature_selector: BorutaFeatureSelector | None = None,
        artifact_store: ArtifactStore | None = None,
        backtester: ForecastBacktester | None = None,
        plotter: ForecastPlotter | None = None,
    ):
        self.config = config or ForecastRunConfig()
        self.feature_engineer = feature_engineer or legacy._FEATURE_ENGINEER
        self.model_registry = model_registry or ForecastModelRegistry()
        self.feature_selector = feature_selector or BorutaFeatureSelector(
            self.config.feature_selection,
            feature_engineer=self.feature_engineer,
        )
        self.artifact_store = artifact_store or ArtifactStore()
        self.backtester = backtester or ForecastBacktester()
        self.plotter = plotter or ForecastPlotter()
        legacy._FEATURE_ENGINEER = self.feature_engineer
        self._apply_runtime_config()

    def _apply_runtime_config(self) -> None:
        feature_selection = self.config.feature_selection
        model_competition = self.config.model_competition

        legacy.DEFAULT_HORIZON = self.config.horizon
        legacy.DEFAULT_BACKTEST_STEP_DAYS = self.config.backtest_step_days
        legacy.DEFAULT_MIN_TRAIN_DAYS = self.config.min_train_days
        legacy.DEFAULT_AUDIT_FOLDS = self.config.audit_folds
        legacy.DEFAULT_INTERVAL_LEVEL = self.config.interval_level
        legacy.DEFAULT_AUDIT_DRIFT_THRESHOLD = self.config.audit_drift_threshold
        legacy.DEFAULT_HYPERPARAM_TRIALS = self.config.hyperparameter_tuning.n_trials
        legacy.DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS = self.config.hyperparameter_tuning.recent_folds
        legacy.DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP = self.config.hyperparameter_tuning.mae_tie_threshold_pp

        legacy.BASELINE_PROFILE = feature_selection.baseline_profile
        legacy.ENHANCED_PROFILE = feature_selection.enhanced_profile
        legacy.BORUTA_FEATURE_PROFILES = {legacy.ENHANCED_PROFILE, legacy.LEGACY_ENHANCED_PROFILE}
        legacy.DEFAULT_FEATURE_PROFILES = list(feature_selection.default_feature_profiles)
        legacy.CHAIN_BORUTA_ANCHORS = list(feature_selection.chain_boruta_anchors)
        legacy.CHAIN_BORUTA_MIN_ANCHORS = feature_selection.chain_boruta_min_anchors
        legacy.BORUTA_MAX_ITER = feature_selection.boruta_max_iter
        legacy.BORUTA_TREE_COUNT = feature_selection.boruta_tree_count
        legacy.BORUTA_PERC = feature_selection.boruta_perc

        legacy.STATISTICAL_MODELS = list(model_competition.statistical_models)
        legacy.DEFAULT_STATISTICAL_MODELS = list(model_competition.default_statistical_models)
        legacy.ML_RECURSIVE_MODELS = list(model_competition.recursive_models)
        legacy.ML_CHAIN_MODELS = list(model_competition.chain_models)
        legacy.EXPERIMENTAL_MODELS = list(model_competition.experimental_models)
        legacy.DEFAULT_MODELS = list(model_competition.default_models)
        legacy.SUPPORTED_MODELS = list(model_competition.supported_models)
        legacy.MODEL_COMPLEXITY = dict(model_competition.model_complexity)
        legacy.SARIMAX_CANDIDATES = [dict(candidate) for candidate in model_competition.sarimax_candidates]

    def actuals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        self._apply_runtime_config()
        return legacy._actuals(daily_df)

    def forecast_demand(
        self,
        daily_df: pd.DataFrame,
        selected_model: str = "seasonal_naive_7",
        horizon_days: int | None = None,
        interval_quantiles: Optional[dict] = None,
        feature_profile: str = legacy.BASELINE_PROFILE,
        feature_schema: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        self._apply_runtime_config()
        return legacy.forecast_demand(
            daily_df,
            selected_model=selected_model,
            horizon_days=horizon_days or self.config.horizon,
            interval_quantiles=interval_quantiles,
            feature_profile=feature_profile,
            feature_schema=feature_schema,
        )

    def predict_model(
        self,
        history: pd.DataFrame,
        horizon: int,
        model_name: str,
        feature_profile: str = legacy.BASELINE_PROFILE,
        selected_schema: Optional[list[str]] = None,
    ):
        self._apply_runtime_config()
        prediction = algorithm_for_model(model_name).predict(
            history,
            horizon,
            feature_profile=feature_profile,
            selected_schema=selected_schema,
        )
        return prediction.values, prediction.feature_schema

    def run_backtest_detailed(
        self,
        daily_df: pd.DataFrame,
        models: Optional[Iterable[str]] = None,
        horizon: int | None = None,
        scenario_lags: Optional[Iterable[int]] = None,
        min_train_days: int | None = None,
        step_days: int | None = None,
        audit_folds: int | None = None,
        return_feature_artifacts: bool = False,
    ):
        self._apply_runtime_config()
        return legacy.run_backtest_detailed(
            daily_df,
            models=models,
            horizon=horizon or self.config.horizon,
            scenario_lags=scenario_lags,
            min_train_days=min_train_days or self.config.min_train_days,
            step_days=step_days or self.config.backtest_step_days,
            audit_folds=self.config.audit_folds if audit_folds is None else audit_folds,
            return_feature_artifacts=return_feature_artifacts,
        )

    def select_champion(self, overall_metrics: pd.DataFrame, horizon: int, feature_schema: list[str] | None = None) -> ForecastChampion:
        self._apply_runtime_config()
        return legacy.select_champion(overall_metrics, horizon=horizon, feature_schema=feature_schema)

    def run_backtest_and_save(
        self,
        daily_df: pd.DataFrame,
        paths: dict,
        horizon: int | None = None,
        scenario_lags: Optional[Iterable[int]] = None,
        models: Optional[Iterable[str]] = None,
    ):
        self._apply_runtime_config()
        return legacy.run_backtest_and_save(
            daily_df,
            paths=paths,
            horizon=horizon or self.config.horizon,
            scenario_lags=scenario_lags,
            models=models,
        )

    def run_forecast_and_save(self, daily_df: pd.DataFrame, paths: dict, horizon: int | None = None):
        self._apply_runtime_config()
        return legacy.run_forecast_and_save(daily_df, paths=paths, horizon=horizon or self.config.horizon)
