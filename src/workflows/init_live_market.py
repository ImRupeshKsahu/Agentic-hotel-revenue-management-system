import argparse

import pandas as pd

from config import BASE_CAPACITY, DATA_END_DATE, FORECAST_OUTPUT_PATH, LIVE_DATA_PATH, RAW_BOOKINGS_PATH
from market_core.feed import ensure_competitor_market
from pms_core.snapshot import calculate_otb_snapshot, export_live_market_state, load_booking_ledger
from pms_core.live_ledger import initialize_live_ledger


def initialize_market_file():
    initialize_live_ledger(output_file=LIVE_DATA_PATH, raw_path=RAW_BOOKINGS_PATH, as_of_date=DATA_END_DATE)
    bookings = load_booking_ledger(LIVE_DATA_PATH)
    snapshot = calculate_otb_snapshot(bookings, as_of_date=DATA_END_DATE, horizon_days=30, capacity=BASE_CAPACITY)

    competitor_rates = None
    try:
        competitor_rates = pd.read_csv(FORECAST_OUTPUT_PATH)
        competitor_rates["Date"] = pd.to_datetime(competitor_rates["Date"])
    except FileNotFoundError:
        competitor_rates = None

    market_snapshot = ensure_competitor_market(
        snapshot["Date"],
        baseline_rates=competitor_rates,
        as_of_timestamp=DATA_END_DATE,
    )
    export_live_market_state(
        snapshot,
        competitor_rates=competitor_rates,
        market_snapshots=market_snapshot,
        output_path="data/live_market_state.json",
    )
    print("live_market_state.json exported from PMS-derived OTB snapshot.")


def main():
    parser = argparse.ArgumentParser(description="Initialize the live PMS ledger, OTB snapshot, and market state.")
    parser.parse_args()
    initialize_market_file()


if __name__ == "__main__":
    main()
