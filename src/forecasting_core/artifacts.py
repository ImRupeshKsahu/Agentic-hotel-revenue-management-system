from __future__ import annotations

import pandas as pd

import forecasting_core.legacy as legacy


ForecastChampion = legacy.ForecastChampion


class ArtifactStore:
    """Forecast artifact persistence with the existing CSV and JSON schemas."""

    def save_champion(self, champion: ForecastChampion, path: str) -> None:
        legacy.save_champion(champion, path)

    def load_champion(self, path: str, default_horizon: int = legacy.DEFAULT_HORIZON) -> ForecastChampion:
        return legacy.load_champion(path, default_horizon=default_horizon)

    def safe_to_csv(self, df: pd.DataFrame, path: str, index: bool = False) -> str:
        return legacy._safe_to_csv(df, path, index=index)

    def feature_manifest(
        self,
        history: pd.DataFrame,
        horizon: int,
        champion_profile: str,
        champion_schema,
    ) -> pd.DataFrame:
        return legacy._feature_manifest(history, horizon, champion_profile, champion_schema)
