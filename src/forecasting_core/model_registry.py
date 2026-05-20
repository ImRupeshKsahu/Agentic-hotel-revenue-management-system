from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import forecasting_core.legacy as legacy


STATISTICAL_MODELS = legacy.STATISTICAL_MODELS
DEFAULT_STATISTICAL_MODELS = legacy.DEFAULT_STATISTICAL_MODELS
ML_RECURSIVE_MODELS = legacy.ML_RECURSIVE_MODELS
ML_CHAIN_MODELS = legacy.ML_CHAIN_MODELS
EXPERIMENTAL_MODELS = legacy.EXPERIMENTAL_MODELS
SUPPORTED_MODELS = legacy.SUPPORTED_MODELS
DEFAULT_MODELS = legacy.DEFAULT_MODELS
MODEL_COMPLEXITY = legacy.MODEL_COMPLEXITY
SARIMAX_CANDIDATES = legacy.SARIMAX_CANDIDATES


class ForecastModelRegistry:
    """Central place for model lists, estimator construction, and availability."""

    @property
    def statistical_models(self) -> list[str]:
        return list(legacy.STATISTICAL_MODELS)

    @property
    def default_models(self) -> list[str]:
        return list(legacy.DEFAULT_MODELS)

    @property
    def supported_models(self) -> list[str]:
        return list(legacy.SUPPORTED_MODELS)

    @property
    def complexity(self) -> dict[str, int]:
        return dict(legacy.MODEL_COMPLEXITY)

    def strategy_for_model(self, model_name: str) -> str:
        return legacy._strategy_for_model(model_name)

    def base_estimator(self, model_name: str):
        return legacy._base_estimator(model_name)

    def unavailable_model_reason(self, model_name: str) -> Optional[str]:
        return legacy._unavailable_model_reason(model_name)

    def available_models(self, models: Iterable[str]) -> list[str]:
        return legacy.available_models(models)
