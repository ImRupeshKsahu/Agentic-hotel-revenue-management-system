from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    optuna = None


TUNING_OBJECTIVE = "recent_cv_mae_pp_rmse_guardrail"
TUNING_MAE_TIE_THRESHOLD_PP = 0.15


@dataclass(frozen=True)
class HyperparameterTuningConfig:
    """Small Optuna budget used for production-style ML model tuning."""

    n_trials: int = 5
    recent_folds: int = 5
    objective: str = TUNING_OBJECTIVE
    mae_tie_threshold_pp: float = TUNING_MAE_TIE_THRESHOLD_PP


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tunable_models(models: Iterable[str]) -> list[str]:
    import forecasting_core.legacy as legacy

    tunable_prefixes = ("random_forest_", "extra_trees_", "xgboost_")
    return [
        model_name
        for model_name in models
        if model_name.startswith(tunable_prefixes)
        and model_name not in legacy.STATISTICAL_MODELS
        and legacy._unavailable_model_reason(model_name) is None
    ]


def _artifact_metadata(
    history: pd.DataFrame,
    models: Iterable[str],
    horizon: int,
    config: HyperparameterTuningConfig,
) -> dict:
    import forecasting_core.legacy as legacy

    max_date = pd.to_datetime(history["Date"].max()).date().isoformat() if not history.empty else None
    return {
        "objective": config.objective,
        "n_trials": int(config.n_trials),
        "recent_folds": int(config.recent_folds),
        "mae_tie_threshold_pp": float(config.mae_tie_threshold_pp),
        "horizon": int(horizon),
        "models": sorted(_tunable_models(models)),
        "data_end_date": max_date,
        "feature_profile": legacy.ENHANCED_PROFILE,
    }


def tuning_artifact_is_current(
    payload: dict,
    history: pd.DataFrame,
    models: Iterable[str],
    horizon: int,
    config: HyperparameterTuningConfig,
) -> bool:
    expected_metadata = _artifact_metadata(history, models, horizon, config)
    if dict(payload.get("_metadata", {})) != expected_metadata:
        return False
    if expected_metadata["models"] and payload.get("_status") != "ok":
        return False
    return True


def load_tuning_payload(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def tuned_params_from_payload(payload: dict) -> dict[str, dict]:
    params = {}
    for model_name, model_payload in payload.items():
        if model_name.startswith("_") or not isinstance(model_payload, dict):
            continue
        best_params = model_payload.get("best_params", {})
        if isinstance(best_params, dict):
            params[model_name] = best_params
    return params


def save_tuning_payload(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, default=str)


class ForecastHyperparameterTuner:
    """Tune ML model params once from the full available history."""

    def __init__(self, config: Optional[HyperparameterTuningConfig] = None):
        self.config = config or HyperparameterTuningConfig()

    def tune(
        self,
        history: pd.DataFrame,
        models: Iterable[str],
        horizon: int,
    ) -> tuple[dict, pd.DataFrame]:
        import forecasting_core.legacy as legacy

        history = legacy._actuals(history)
        model_names = _tunable_models(models)
        metadata = _artifact_metadata(history, model_names, horizon, self.config)
        payload = {"_metadata": metadata, "_tuned_at": pd.Timestamp.now().isoformat()}
        report_rows = []

        if not model_names:
            return payload, pd.DataFrame(report_rows)

        if optuna is None:
            for model_name in model_names:
                report_rows.append(
                    {
                        "Model": model_name,
                        "Trial": None,
                        "Params": "{}",
                        "MAE_pp": np.nan,
                        "RMSE_pp": np.nan,
                        "Bias_pp": np.nan,
                        "Abs_Bias_pp": np.nan,
                        "WAPE": np.nan,
                        "Folds_Used": 0,
                        "Status": "skipped_optuna_unavailable",
                    }
                )
            payload["_status"] = "skipped_optuna_unavailable"
            return payload, pd.DataFrame(report_rows)

        folds = self._recent_validation_folds(history, horizon)
        if folds.empty:
            for model_name in model_names:
                report_rows.append(
                    {
                        "Model": model_name,
                        "Trial": None,
                        "Params": "{}",
                        "MAE_pp": np.nan,
                        "RMSE_pp": np.nan,
                        "Bias_pp": np.nan,
                        "Abs_Bias_pp": np.nan,
                        "WAPE": np.nan,
                        "Folds_Used": 0,
                        "Status": "skipped_no_recent_folds",
                    }
                )
            payload["_status"] = "skipped_no_recent_folds"
            return payload, pd.DataFrame(report_rows)

        model_specs = legacy._model_specs(model_names)
        production_schemas, _, _ = legacy._select_production_feature_schemas(history, horizon, model_specs)

        for model_name in model_names:
            trial_rows = []

            def objective(trial):
                params = self._suggest_params(trial, model_name)
                try:
                    metrics = self._score_trial(
                        history=history,
                        folds=folds,
                        horizon=horizon,
                        model_name=model_name,
                        params=params,
                        production_schemas=production_schemas,
                    )
                except Exception:
                    metrics = self._failed_trial_metrics()
                score = _safe_float(metrics.get("MAE_pp"), default=np.inf)
                trial_rows.append(
                    {
                        "Model": model_name,
                        "Trial": int(trial.number),
                        "Params": json.dumps(params, sort_keys=True),
                        "MAE_pp": _safe_float(metrics.get("MAE_pp")),
                        "RMSE_pp": _safe_float(metrics.get("RMSE_pp")),
                        "Bias_pp": _safe_float(metrics.get("Bias_pp")),
                        "Abs_Bias_pp": _safe_float(metrics.get("Abs_Bias_pp")),
                        "WAPE": _safe_float(metrics.get("WAPE")),
                        "Folds_Used": int(len(folds)),
                        "Status": "ok" if np.isfinite(score) else "failed",
                    }
                )
                return score if np.isfinite(score) else float("inf")

            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=int(self.config.n_trials), show_progress_bar=False)
            report_rows.extend(trial_rows)

            completed = [row for row in trial_rows if np.isfinite(_safe_float(row["MAE_pp"]))]
            if completed:
                best = self._select_best_trial(completed)
                best_params = json.loads(best["Params"])
                payload[model_name] = {
                    "best_params": best_params,
                    "best_mae_pp": _safe_float(best["MAE_pp"]),
                    "best_rmse_pp": _safe_float(best["RMSE_pp"]),
                    "best_bias_pp": _safe_float(best["Bias_pp"]),
                    "best_abs_bias_pp": _safe_float(best["Abs_Bias_pp"]),
                    "best_wape": _safe_float(best["WAPE"]),
                    "n_trials": int(self.config.n_trials),
                    "objective": self.config.objective,
                    "tuning_mae_tie_threshold_pp": float(self.config.mae_tie_threshold_pp),
                    "recent_folds_used": int(len(folds)),
                    "data_end_date": metadata["data_end_date"],
                    "strategy": legacy._strategy_for_model(model_name),
                    "tuned_at": payload["_tuned_at"],
                }
            else:
                payload[model_name] = {
                    "best_params": {},
                    "best_mae_pp": np.nan,
                    "best_rmse_pp": np.nan,
                    "best_bias_pp": np.nan,
                    "best_abs_bias_pp": np.nan,
                    "best_wape": np.nan,
                    "n_trials": int(self.config.n_trials),
                    "objective": self.config.objective,
                    "tuning_mae_tie_threshold_pp": float(self.config.mae_tie_threshold_pp),
                    "recent_folds_used": int(len(folds)),
                    "data_end_date": metadata["data_end_date"],
                    "strategy": legacy._strategy_for_model(model_name),
                    "tuned_at": payload["_tuned_at"],
                    "status": "failed_all_trials",
                }

        payload["_status"] = "ok"
        return payload, pd.DataFrame(report_rows)

    def _recent_validation_folds(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        import forecasting_core.legacy as legacy

        folds = legacy._generate_weekly_folds(
            history,
            horizon,
            min_train_days=legacy.DEFAULT_MIN_TRAIN_DAYS,
            step_days=legacy.DEFAULT_BACKTEST_STEP_DAYS,
            audit_folds=0,
        )
        if folds.empty:
            return folds
        return folds.tail(int(self.config.recent_folds)).reset_index(drop=True)

    def _score_trial(
        self,
        history: pd.DataFrame,
        folds: pd.DataFrame,
        horizon: int,
        model_name: str,
        params: dict,
        production_schemas: dict[str, list[str]],
    ) -> dict:
        import forecasting_core.legacy as legacy

        rows = []
        strategy = legacy._strategy_for_model(model_name)
        selected_schema = production_schemas.get(strategy)
        for fold in folds.itertuples(index=False):
            cutoff = pd.to_datetime(fold.Cutoff)
            train = history[history["Date"].le(cutoff)].copy()
            actual_future = history[
                (history["Date"].gt(cutoff))
                & (history["Date"].le(cutoff + pd.Timedelta(days=horizon)))
            ].copy()
            if len(actual_future) != horizon:
                continue
            preds, _ = legacy.predict_model(
                train,
                horizon,
                model_name,
                feature_profile=legacy.ENHANCED_PROFILE,
                selected_schema=selected_schema,
                tuned_params=params,
            )
            valid_len = min(len(actual_future), len(preds))
            for idx in range(valid_len):
                rows.append(
                    {
                        "Actual": float(actual_future.iloc[idx]["Occupancy_Rate"]),
                        "Predicted": float(preds[idx]),
                    }
                )
        if not rows:
            return self._failed_trial_metrics()
        frame = pd.DataFrame(rows)
        metrics = legacy.calculate_forecast_metrics(frame["Actual"].to_numpy(), frame["Predicted"].to_numpy())
        return {
            "MAE_pp": _safe_float(metrics.get("MAE_pp"), default=np.inf),
            "RMSE_pp": _safe_float(metrics.get("RMSE_pp"), default=np.inf),
            "Bias_pp": _safe_float(metrics.get("Bias_pp")),
            "Abs_Bias_pp": _safe_float(metrics.get("Abs_Bias_pp"), default=np.inf),
            "WAPE": _safe_float(metrics.get("WAPE")),
        }

    def _failed_trial_metrics(self) -> dict:
        return {
            "MAE_pp": np.inf,
            "RMSE_pp": np.inf,
            "Bias_pp": np.nan,
            "Abs_Bias_pp": np.inf,
            "WAPE": np.nan,
        }

    def _select_best_trial(self, rows: list[dict]) -> dict:
        best_mae = min(_safe_float(row["MAE_pp"], default=np.inf) for row in rows)
        eligible = [
            row
            for row in rows
            if _safe_float(row["MAE_pp"], default=np.inf) <= best_mae + float(self.config.mae_tie_threshold_pp)
        ]
        return min(
            eligible,
            key=lambda row: (
                _safe_float(row["RMSE_pp"], default=np.inf),
                _safe_float(row["MAE_pp"], default=np.inf),
                _safe_float(row["Abs_Bias_pp"], default=np.inf),
                int(row.get("Trial", 0) or 0),
            ),
        )

    def _suggest_params(self, trial, model_name: str) -> dict:
        if model_name.startswith("random_forest"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 12, 60),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        if model_name.startswith("extra_trees"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 16, 80),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        if model_name.startswith("xgboost"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 8, 48),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
                "subsample": trial.suggest_float("subsample", 0.70, 1.00),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }
        return {}
