import argparse
import os

import pandas as pd

from project_core.config import (
    BACKTEST_AUDIT_FOLD_METRICS_PATH,
    BACKTEST_AUDIT_INTERVAL_COVERAGE_PATH,
    BACKTEST_AUDIT_LAG_METRICS_PATH,
    BACKTEST_AUDIT_PREDICTIONS_PATH,
    BACKTEST_AUDIT_SUMMARY_PATH,
    BORUTA_SELECTION_REPORT_PATH,
    BACKTEST_CADENCE_DAYS,
    BACKTEST_FOLD_METRICS_PATH,
    BACKTEST_LAG_METRICS_PATH,
    BACKTEST_PREDICTIONS_PATH,
    BACKTEST_SCENARIO_METRICS_PATH,
    BACKTEST_TIMELINE_PATH,
    BASE_CAPACITY,
    DATA_END_DATE,
    DATA_PATH,
    FORECAST_CHAMPION_PATH,
    FORECAST_HORIZON_DAYS,
    FORECAST_OUTPUT_PATH,
    LIVE_COMPETITOR_MARKET_PATH,
    LIVE_DATA_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_MANIFEST_PATH,
    FORECAST_HYPERPARAM_TRIALS,
    FORECAST_HYPERPARAM_TUNING_PATH,
    FORECAST_HYPERPARAM_TUNING_RECENT_FOLDS,
    FORECAST_HYPERPARAM_TUNING_REPORT_PATH,
    OTB_SNAPSHOT_PATH,
    PLOTS_DIR,
    RAW_BOOKINGS_PATH,
)
from pms_core.data_pipeline import refresh_daily_hotel_data
from forecasting_core.api import ForecastEngine, load_champion
from forecasting_core.hyperparameter_tuning import (
    HyperparameterTuningConfig,
    load_tuning_payload,
    tuning_artifact_is_current,
)
from market_core.feed import initialize_competitor_market
from pms_core.snapshot import calculate_otb_snapshot, export_live_market_state, load_booking_ledger
from pms_core.live_ledger import initialize_live_ledger


_FORECAST_ENGINE = ForecastEngine()


def _safe_to_csv(df, path, index=False):
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(fallback, index=index)
        print(f"Could not overwrite locked file {path}. Saved copy to {fallback}")
        return fallback


def _artifact_paths():
    return {
        "forecast": FORECAST_OUTPUT_PATH,
        "metrics": METRICS_PATH,
        "comparison": MODEL_COMPARISON_PATH,
        "lag_metrics": BACKTEST_LAG_METRICS_PATH,
        "scenario_metrics": BACKTEST_SCENARIO_METRICS_PATH,
        "predictions": BACKTEST_PREDICTIONS_PATH,
        "fold_metrics": BACKTEST_FOLD_METRICS_PATH,
        "audit_predictions": BACKTEST_AUDIT_PREDICTIONS_PATH,
        "audit_fold_metrics": BACKTEST_AUDIT_FOLD_METRICS_PATH,
        "audit_summary": BACKTEST_AUDIT_SUMMARY_PATH,
        "audit_lag_metrics": BACKTEST_AUDIT_LAG_METRICS_PATH,
        "audit_interval_coverage": BACKTEST_AUDIT_INTERVAL_COVERAGE_PATH,
        "feature_manifest": FEATURE_MANIFEST_PATH,
        "boruta_selection_report": BORUTA_SELECTION_REPORT_PATH,
        "hyperparameter_tuning": FORECAST_HYPERPARAM_TUNING_PATH,
        "hyperparameter_tuning_report": FORECAST_HYPERPARAM_TUNING_REPORT_PATH,
        "champion": FORECAST_CHAMPION_PATH,
        "plots_dir": PLOTS_DIR,
        "timeline_plot": BACKTEST_TIMELINE_PATH,
    }


def _refresh_daily_data():
    print("=" * 60)
    print("REFRESHING PMS-DERIVED DAILY DATA")
    print("=" * 60)
    return refresh_daily_hotel_data(
        DATA_PATH,
        raw_path=RAW_BOOKINGS_PATH,
        capacity=BASE_CAPACITY,
        as_of_date=DATA_END_DATE,
        horizon_days=FORECAST_HORIZON_DAYS,
    )


def _seed_live_pms_and_otb(forecast_df):
    print("\n" + "=" * 60)
    print("SEEDING LIVE PMS LEDGER AND OTB SNAPSHOT")
    print("=" * 60)
    initialize_live_ledger(output_file=LIVE_DATA_PATH, raw_path=RAW_BOOKINGS_PATH, as_of_date=DATA_END_DATE)
    bookings = load_booking_ledger(LIVE_DATA_PATH)
    otb_snapshot = calculate_otb_snapshot(
        bookings,
        as_of_date=DATA_END_DATE,
        horizon_days=FORECAST_HORIZON_DAYS,
        capacity=BASE_CAPACITY,
    )
    _safe_to_csv(otb_snapshot, OTB_SNAPSHOT_PATH, index=False)
    market_snapshot = initialize_competitor_market(
        otb_snapshot["Date"],
        baseline_rates=forecast_df,
        as_of_timestamp=DATA_END_DATE,
        output_path=LIVE_COMPETITOR_MARKET_PATH,
    )
    export_live_market_state(
        otb_snapshot,
        competitor_rates=forecast_df,
        market_snapshots=market_snapshot,
        output_path="data/live_market_state.json",
    )


def run_backtest_mode():
    daily_df = _refresh_daily_data()
    print("\n" + "=" * 60)
    print("RUNNING WEEKLY MODEL COMPETITION BACKTEST")
    print("=" * 60)
    forecast_df, metrics_df, champion = _FORECAST_ENGINE.run_backtest_and_save(
        daily_df,
        paths=_artifact_paths(),
        horizon=FORECAST_HORIZON_DAYS,
    )
    _seed_live_pms_and_otb(forecast_df)
    _print_summary(forecast_df, metrics_df, champion, mode="backtest")
    return forecast_df, metrics_df, champion


def run_forecast_mode():
    daily_df = _refresh_daily_data()
    if not os.path.exists(FORECAST_CHAMPION_PATH):
        print("No forecast champion found. Running backtest mode first to select a model.")
        return run_backtest_mode()
    if not _hyperparameter_tuning_is_current(daily_df):
        print("Hyperparameter tuning artifact is missing or stale. Running backtest mode first to refresh tuned params.")
        return run_backtest_mode()

    print("\n" + "=" * 60)
    print("RUNNING DAILY CHAMPION FORECAST")
    print("=" * 60)
    forecast_df, champion = _FORECAST_ENGINE.run_forecast_and_save(
        daily_df,
        paths=_artifact_paths(),
        horizon=FORECAST_HORIZON_DAYS,
    )
    metrics_df = pd.read_csv(MODEL_COMPARISON_PATH) if os.path.exists(MODEL_COMPARISON_PATH) else pd.DataFrame()
    _seed_live_pms_and_otb(forecast_df)
    _print_summary(forecast_df, metrics_df, champion, mode="forecast")
    return forecast_df, metrics_df, champion


def _hyperparameter_tuning_is_current(daily_df):
    payload = load_tuning_payload(FORECAST_HYPERPARAM_TUNING_PATH)
    if not payload:
        return False
    history = _FORECAST_ENGINE.actuals(daily_df)
    config = HyperparameterTuningConfig(
        n_trials=FORECAST_HYPERPARAM_TRIALS,
        recent_folds=FORECAST_HYPERPARAM_TUNING_RECENT_FOLDS,
    )
    return tuning_artifact_is_current(
        payload,
        history,
        _FORECAST_ENGINE.config.model_competition.default_models,
        FORECAST_HORIZON_DAYS,
        config,
    )


def _print_summary(forecast_df, metrics_df, champion, mode):
    print(f"Mode: {mode}")
    print(f"Daily data saved to {DATA_PATH}")
    print(f"Live PMS ledger saved to {LIVE_DATA_PATH}")
    print(f"OTB snapshot saved to {OTB_SNAPSHOT_PATH}")
    print(f"Forecast saved to {FORECAST_OUTPUT_PATH}")
    print(f"Champion metadata saved to {FORECAST_CHAMPION_PATH}")
    print(f"Forecast plots saved to {PLOTS_DIR}")
    print(f"Selected champion: {champion.model} ({champion.strategy})")
    if not metrics_df.empty:
        print("\nTop model rows:")
        print(metrics_df.head(5).to_string(index=False))


def backtest_is_stale(champion_path=FORECAST_CHAMPION_PATH, cadence_days=BACKTEST_CADENCE_DAYS):
    if not os.path.exists(champion_path):
        return True
    champion = load_champion(champion_path, default_horizon=FORECAST_HORIZON_DAYS)
    selected_at = pd.to_datetime(champion.selected_at, errors="coerce")
    if pd.isna(selected_at):
        return True
    return pd.Timestamp.now(tz=selected_at.tz) - selected_at > pd.Timedelta(days=cadence_days)


def main():
    parser = argparse.ArgumentParser(description="Hotel RMS occupancy forecasting pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["backtest", "forecast", "auto"],
        default="auto",
        help="backtest runs model competition; forecast uses the saved champion; auto backtests only when stale/missing.",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest_mode()
    elif args.mode == "forecast":
        run_forecast_mode()
    elif backtest_is_stale():
        print(f"Backtest is missing or older than {BACKTEST_CADENCE_DAYS} days. Running backtest mode.")
        run_backtest_mode()
    else:
        run_forecast_mode()


if __name__ == "__main__":
    main()
