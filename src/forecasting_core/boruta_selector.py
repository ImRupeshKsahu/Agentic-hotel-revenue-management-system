from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from forecasting_core.config import FeatureSelectionConfig
import forecasting_core.legacy as legacy


class BorutaFeatureSelector:
    """Feature-selection service for enhanced forecasting profiles."""

    def __init__(self, config: FeatureSelectionConfig | None = None, feature_engineer=None):
        self.config = config or FeatureSelectionConfig()
        self.feature_engineer = feature_engineer or legacy._FEATURE_ENGINEER

    def available(self) -> bool:
        return legacy._boruta_available()

    def unavailable_reason(self) -> Optional[str]:
        return legacy._boruta_unavailable_reason()

    def force_keep_features(self, columns: Iterable[str]) -> list[str]:
        return legacy._force_keep_features(columns)

    def run(self, x_train: pd.DataFrame, y_train: np.ndarray, anchor: str) -> pd.DataFrame:
        return legacy._run_boruta(x_train, y_train, anchor=anchor)

    def stable_features_from_report(self, report: pd.DataFrame, min_anchor_count: int) -> tuple[list[str], str]:
        return legacy._stable_features_from_boruta(report, min_anchor_count)

    def select_recursive_schema(self, x_train: pd.DataFrame, y_train: np.ndarray) -> tuple[list[str], pd.DataFrame, dict]:
        return legacy._select_recursive_schema(x_train, y_train)

    def select_chain_schema(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        anchors: Iterable[int] | None = None,
    ) -> tuple[list[str], pd.DataFrame, dict]:
        return legacy._select_chain_schema(
            x_train,
            y_train,
            anchors=list(anchors or self.config.chain_boruta_anchors),
        )

    def select_production_feature_schemas(
        self,
        history: pd.DataFrame,
        horizon: int,
        model_specs: Iterable[dict],
    ) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
        return legacy._select_production_feature_schemas(history, horizon, model_specs)
