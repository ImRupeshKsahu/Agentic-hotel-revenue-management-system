import os
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Base Directory (The root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if load_dotenv:
    load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_PATH = os.path.join(BASE_DIR, "data", "daily_hotel_data.csv")
RAW_BOOKINGS_PATH = os.path.join(BASE_DIR, "data", "hotel_bookings.csv")
OTB_SNAPSHOT_PATH = os.path.join(BASE_DIR, "data", "otb_snapshot.csv")
LIVE_MARKET_STATE_PATH = os.path.join(BASE_DIR, "data", "live_market_state.json")
LIVE_COMPETITOR_MARKET_PATH = os.path.join(BASE_DIR, "data", "live_competitor_market.csv")
LOCAL_INTEL_CALENDAR_PATH = os.path.join(BASE_DIR, "data", "local_intel_calendar.csv")
PRICING_DECISION_LOG_PATH = os.path.join(BASE_DIR, "data", "pricing_decision_log.jsonl")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "data", "model_comparison_metrics.csv")
BACKTEST_LAG_METRICS_PATH = os.path.join(BASE_DIR, "data", "backtest_lag_metrics.csv")
BACKTEST_SCENARIO_METRICS_PATH = os.path.join(BASE_DIR, "data", "backtest_scenario_metrics.csv")
BACKTEST_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "backtest_predictions.csv")
BACKTEST_FOLD_METRICS_PATH = os.path.join(BASE_DIR, "data", "backtest_fold_metrics.csv")
BACKTEST_AUDIT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "backtest_audit_predictions.csv")
BACKTEST_AUDIT_FOLD_METRICS_PATH = os.path.join(BASE_DIR, "data", "backtest_audit_fold_metrics.csv")
BACKTEST_AUDIT_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "backtest_audit_summary.csv")
BACKTEST_AUDIT_LAG_METRICS_PATH = os.path.join(BASE_DIR, "data", "backtest_audit_lag_metrics.csv")
BACKTEST_AUDIT_INTERVAL_COVERAGE_PATH = os.path.join(BASE_DIR, "data", "backtest_audit_interval_coverage.csv")
FEATURE_MANIFEST_PATH = os.path.join(BASE_DIR, "data", "feature_manifest.csv")
BORUTA_SELECTION_REPORT_PATH = os.path.join(BASE_DIR, "data", "boruta_selection_report.csv")
FORECAST_HYPERPARAM_TUNING_PATH = os.path.join(BASE_DIR, "data", "model_hyperparameters.json")
FORECAST_HYPERPARAM_TUNING_REPORT_PATH = os.path.join(BASE_DIR, "data", "hyperparameter_tuning_report.csv")
FORECAST_CHAMPION_PATH = os.path.join(BASE_DIR, "data", "forecast_champion.json")
PLOTS_DIR = os.path.join(BASE_DIR, "data", "plots")
BACKTEST_TIMELINE_PATH = os.path.join(BASE_DIR, "docs", "backtest_timeline_explainer.png")
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "prophet_model.pkl")
FORECAST_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "demand_forecast_output.csv")
METRICS_PATH= os.path.join(BASE_DIR, "data", "model_validation_metrics.csv")
BASE_PRICE = 100  
BASE_CAPACITY = 237 # integer capacity
DEFAULT_HOTEL = "City Hotel"
MIN_PRICE = 80
MAX_PRICE = 250
PRICE_STEP = 5
BASE_RATE_LOOKBACK_DAYS = 90
DYNAMIC_BASE_DOW_WEIGHT = 0.70
DYNAMIC_FLOOR_COMP_LOW_FACTOR = 0.95
DYNAMIC_CEILING_BASE_MULTIPLIER = 1.15
DEFAULT_EVENT_IMPACT = 0.20 # 20% demand boost
API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATA_END_DATE = pd.to_datetime("2017-08-31")
FORECAST_HORIZON_DAYS = 30
BACKTEST_SCENARIO_LAGS = [10, 14, 21, 30, 45, 60]
BACKTEST_CADENCE_DAYS = 7
FORECAST_HYPERPARAM_TRIALS = 5
FORECAST_HYPERPARAM_TUNING_RECENT_FOLDS = 5
STRATEGIST_PROMPT_PATH = os.path.join(BASE_DIR,"src", "prompts", "strategist.txt")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "hotel_bookings.csv")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "live_hotel_bookings.csv")

# Forecast feature engineering definitions.
# Keep the feature contract here so feature additions/removals are reviewable without
# digging through model code.
FORECAST_BASELINE_LAGS = [1, 2, 3, 7, 14, 21, 28, 56]
FORECAST_ENHANCED_EXTRA_LAGS = [4, 5, 6, 35, 42, 84, 112, 364]
FORECAST_ROLLING_WINDOWS = [7, 14, 28, 56]
FORECAST_BASELINE_ROLLING_STATS = ["mean", "std"]
FORECAST_ENHANCED_ROLLING_STATS = ["min", "max", "slope"]
FORECAST_BASELINE_TREND_FEATURES = {
    "trend_7": ("roll_mean_7", "roll_mean_28"),
    "trend_14": ("roll_mean_14", "roll_mean_56"),
}
FORECAST_TREND_PROJECTION_FEATURES = {
    "trend_projection_7_from_14": ("roll_mean_14", "roll_slope_14", 7),
    "trend_projection_14_from_28": ("roll_mean_28", "roll_slope_28", 14),
    "trend_projection_30_from_56": ("roll_mean_56", "roll_slope_56", 30),
}
FORECAST_RECENT_EXTREME_WINDOW = 28
FORECAST_RECENT_OPERATIONAL_WINDOW = 14
FORECAST_ORIGIN_CALENDAR_FEATURES = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_weekend"]
FORECAST_FUTURE_KNOWN_FEATURES = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_weekend", "local_event"]
FORECAST_SEASONAL_INDEX_FEATURES = ["dow_seasonal_index", "month_seasonal_index"]
FORECAST_OPERATIONAL_SIGNAL_COLUMNS = {
    "booking_pace": "Booking_Pace",
    "bookings_created": "Bookings_Created",
    "cancellations": "Cancellations",
}
FORECAST_OPERATIONAL_WINDOWS = [7, 14, 28]
FORECAST_OPERATIONAL_STATS = ["mean", "std", "sum"]

# Enhanced Boruta is allowed to choose from newly engineered features, but these
# proven baseline features are always kept as the model's stable forecasting spine.
FORECAST_FORCE_KEEP_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_56",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_mean_28",
    "roll_std_28",
    "roll_mean_56",
    "roll_std_56",
    "trend_7",
    "trend_14",
    "recent_min_28",
    "recent_max_28",
    "recent_booking_pace",
    "recent_cancellations",
    "origin_dow_sin",
    "origin_dow_cos",
    "origin_doy_sin",
    "origin_doy_cos",
    "origin_month_sin",
    "origin_month_cos",
    "origin_is_weekend",
]
