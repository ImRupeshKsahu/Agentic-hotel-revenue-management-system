from __future__ import annotations

from dataclasses import dataclass, field

import forecasting_core.legacy as legacy
from forecasting_core.hyperparameter_tuning import HyperparameterTuningConfig


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Tunable feature-selection knobs layered over the existing constants."""

    baseline_profile: str = legacy.BASELINE_PROFILE
    enhanced_profile: str = legacy.ENHANCED_PROFILE
    default_feature_profiles: list[str] = field(default_factory=lambda: list(legacy.DEFAULT_FEATURE_PROFILES))
    chain_boruta_anchors: list[int] = field(default_factory=lambda: list(legacy.CHAIN_BORUTA_ANCHORS))
    chain_boruta_min_anchors: int = legacy.CHAIN_BORUTA_MIN_ANCHORS
    boruta_max_iter: int = legacy.BORUTA_MAX_ITER
    boruta_tree_count: int | str = legacy.BORUTA_TREE_COUNT
    boruta_perc: int = legacy.BORUTA_PERC


@dataclass(frozen=True)
class ModelCompetitionConfig:
    """Model slate and backtest tuning used by weekly model competition."""

    statistical_models: list[str] = field(default_factory=lambda: list(legacy.STATISTICAL_MODELS))
    default_statistical_models: list[str] = field(default_factory=lambda: list(legacy.DEFAULT_STATISTICAL_MODELS))
    recursive_models: list[str] = field(default_factory=lambda: list(legacy.ML_RECURSIVE_MODELS))
    chain_models: list[str] = field(default_factory=lambda: list(legacy.ML_CHAIN_MODELS))
    experimental_models: list[str] = field(default_factory=lambda: list(legacy.EXPERIMENTAL_MODELS))
    default_models: list[str] = field(default_factory=lambda: list(legacy.DEFAULT_MODELS))
    supported_models: list[str] = field(default_factory=lambda: list(legacy.SUPPORTED_MODELS))
    model_complexity: dict[str, int] = field(default_factory=lambda: dict(legacy.MODEL_COMPLEXITY))
    sarimax_candidates: list[dict] = field(default_factory=lambda: [dict(candidate) for candidate in legacy.SARIMAX_CANDIDATES])


@dataclass(frozen=True)
class ForecastRunConfig:
    """Run-level knobs for forecasts, rolling-origin backtests, and audits."""

    horizon: int = legacy.DEFAULT_HORIZON
    scenario_lags: list[int] = field(default_factory=lambda: list(legacy.DEFAULT_SCENARIO_LAGS))
    backtest_step_days: int = legacy.DEFAULT_BACKTEST_STEP_DAYS
    min_train_days: int = legacy.DEFAULT_MIN_TRAIN_DAYS
    audit_folds: int = legacy.DEFAULT_AUDIT_FOLDS
    interval_level: float = legacy.DEFAULT_INTERVAL_LEVEL
    audit_drift_threshold: float = legacy.DEFAULT_AUDIT_DRIFT_THRESHOLD
    hyperparameter_tuning: HyperparameterTuningConfig = field(
        default_factory=lambda: HyperparameterTuningConfig(
            n_trials=legacy.DEFAULT_HYPERPARAM_TRIALS,
            recent_folds=legacy.DEFAULT_HYPERPARAM_TUNING_RECENT_FOLDS,
            mae_tie_threshold_pp=legacy.DEFAULT_HYPERPARAM_TUNING_MAE_TIE_THRESHOLD_PP,
        )
    )
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    model_competition: ModelCompetitionConfig = field(default_factory=ModelCompetitionConfig)
