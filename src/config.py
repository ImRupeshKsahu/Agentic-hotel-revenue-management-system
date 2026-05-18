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
STRATEGIST_PROMPT_PATH = os.path.join(BASE_DIR,"src", "prompts", "strategist.txt")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "hotel_bookings.csv")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "live_hotel_bookings.csv")
