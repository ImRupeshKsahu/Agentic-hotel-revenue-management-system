from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import forecasting_core.legacy as legacy


@dataclass(frozen=True)
class ForecastPrediction:
    """A model forecast plus the feature schema used to produce it."""

    values: np.ndarray
    feature_schema: list[str] = field(default_factory=list)
    fallback_model: str | None = None


class ForecastAlgorithm(ABC):
    """Common interface for statistical, recursive ML, and chain ML models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        feature_profile: str = legacy.BASELINE_PROFILE,
        selected_schema: Optional[list[str]] = None,
    ) -> ForecastPrediction:
        raise NotImplementedError


class StatisticalAlgorithm(ForecastAlgorithm):
    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        feature_profile: str = legacy.BASELINE_PROFILE,
        selected_schema: Optional[list[str]] = None,
    ) -> ForecastPrediction:
        values = legacy._predict_statistical(history, horizon, self.model_name)
        return ForecastPrediction(values=np.clip(values, 0, 1), feature_schema=[])


class RecursiveMLAlgorithm(ForecastAlgorithm):
    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        feature_profile: str = legacy.BASELINE_PROFILE,
        selected_schema: Optional[list[str]] = None,
    ) -> ForecastPrediction:
        values, schema = legacy._predict_recursive(
            history,
            horizon,
            self.model_name,
            feature_profile=feature_profile,
            selected_schema=selected_schema,
        )
        return ForecastPrediction(values=np.clip(values, 0, 1), feature_schema=schema)


class ChainMLAlgorithm(ForecastAlgorithm):
    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        feature_profile: str = legacy.BASELINE_PROFILE,
        selected_schema: Optional[list[str]] = None,
    ) -> ForecastPrediction:
        values, schema = legacy._predict_chain(
            history,
            horizon,
            self.model_name,
            feature_profile=feature_profile,
            selected_schema=selected_schema,
        )
        return ForecastPrediction(values=np.clip(values, 0, 1), feature_schema=schema)


def algorithm_for_model(model_name: str) -> ForecastAlgorithm:
    if model_name in legacy.STATISTICAL_MODELS:
        return StatisticalAlgorithm(model_name)
    if model_name.endswith("_chain"):
        return ChainMLAlgorithm(model_name)
    if model_name.endswith("_recursive"):
        return RecursiveMLAlgorithm(model_name)
    return StatisticalAlgorithm("rolling_mean_7")
