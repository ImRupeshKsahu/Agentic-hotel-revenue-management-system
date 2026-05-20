import argparse
import os

import pandas as pd

from config import DATA_END_DATE, FORECAST_HORIZON_DAYS, FORECAST_OUTPUT_PATH, LIVE_COMPETITOR_MARKET_PATH, OTB_SNAPSHOT_PATH
from market_core.feed import initialize_competitor_market, simulate_competitor_market_event
from pms_core.snapshot import export_live_market_state


def _forecast_rates():
    try:
        rates = pd.read_csv(FORECAST_OUTPUT_PATH)
        rates["Date"] = pd.to_datetime(rates["Date"])
        return rates
    except FileNotFoundError:
        return None


def _stay_dates():
    return pd.date_range(pd.Timestamp(DATA_END_DATE) + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq="D")


def initialize_market():
    snapshot = initialize_competitor_market(
        _stay_dates(),
        baseline_rates=_forecast_rates(),
        as_of_timestamp=DATA_END_DATE,
        output_path=LIVE_COMPETITOR_MARKET_PATH,
    )
    refresh_live_market_state(snapshot)
    return snapshot


def refresh_live_market_state(market_snapshot=None):
    if not os.path.exists(OTB_SNAPSHOT_PATH):
        return None
    otb_snapshot = pd.read_csv(OTB_SNAPSHOT_PATH)
    otb_snapshot["Date"] = pd.to_datetime(otb_snapshot["Date"])
    export_live_market_state(otb_snapshot, market_snapshots=market_snapshot, output_path="data/live_market_state.json")
    return otb_snapshot


def simulate_once(seed=None):
    event = simulate_competitor_market_event(
        stay_dates=_stay_dates(),
        baseline_rates=_forecast_rates(),
        as_of_timestamp=DATA_END_DATE,
        path=LIVE_COMPETITOR_MARKET_PATH,
        seed=seed,
    )
    market_snapshot = pd.read_csv(LIVE_COMPETITOR_MARKET_PATH)
    refresh_live_market_state(market_snapshot)
    return event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate external competitor market observations for the Hotel RMS demo.")
    parser.add_argument("--init", action="store_true", help="Initialize the live competitor market snapshot and exit.")
    parser.add_argument("--once", action="store_true", help="Simulate one competitor market move and exit.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible one-off market moves.")
    args = parser.parse_args()

    if args.init:
        frame = initialize_market()
        print(f"Initialized {LIVE_COMPETITOR_MARKET_PATH} with {len(frame)} market rows.")
    elif args.once:
        print(simulate_once(seed=args.seed))
    else:
        parser.error("Choose --init or --once.")
