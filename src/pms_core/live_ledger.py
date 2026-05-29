import argparse
import os
import random
import time

import pandas as pd

from project_core.config import BASE_CAPACITY, DATA_END_DATE, DEFAULT_HOTEL, FORECAST_OUTPUT_PATH, LIVE_DATA_PATH, LIVE_MARKET_STATE_PATH, OTB_SNAPSHOT_PATH, RAW_BOOKINGS_PATH
from pms_core.data_pipeline import normalize_bookings
from market_core.feed import ensure_competitor_market
from pms_core.snapshot import calculate_otb_snapshot, export_live_market_state


def refresh_live_market_artifacts(ledger, as_of_date=DATA_END_DATE):
    """Refresh small Streamlit-facing artifacts after the PMS ledger changes."""
    snapshot = calculate_otb_snapshot(
        ledger,
        as_of_date=as_of_date,
        horizon_days=30,
        capacity=BASE_CAPACITY,
    )
    snapshot.to_csv(OTB_SNAPSHOT_PATH, index=False)

    competitor_rates = None
    try:
        competitor_rates = pd.read_csv(FORECAST_OUTPUT_PATH)
        competitor_rates["Date"] = pd.to_datetime(competitor_rates["Date"])
    except FileNotFoundError:
        competitor_rates = None

    market_snapshot = ensure_competitor_market(
        snapshot["Date"],
        baseline_rates=competitor_rates,
        as_of_timestamp=as_of_date,
    )
    export_live_market_state(
        snapshot,
        competitor_rates=competitor_rates,
        market_snapshots=market_snapshot,
        output_path=LIVE_MARKET_STATE_PATH,
    )
    return snapshot


def _synthetic_future_on_books(as_of_date, seed=42):
    rng = random.Random(seed)
    try:
        forecast = pd.read_csv(FORECAST_OUTPUT_PATH)
        forecast["Date"] = pd.to_datetime(forecast["Date"])
    except FileNotFoundError:
        forecast = pd.DataFrame(
            {
                "Date": pd.date_range(pd.to_datetime(as_of_date) + pd.Timedelta(days=1), periods=30, freq="D"),
                "Forecasted_Occupancy": 0.55,
                "Competitor_Rate": 125.0,
            }
        )

    rows = []
    next_id = 10_000_000
    as_of_date = pd.to_datetime(as_of_date)
    for row in forecast.itertuples(index=False):
        stay_date = pd.to_datetime(row.Date)
        final_occ = float(getattr(row, "Forecasted_Occupancy", 0.55))
        rooms_on_books = int(BASE_CAPACITY * min(max(final_occ * rng.uniform(0.35, 0.72), 0.05), 0.85))
        adr = float(getattr(row, "Competitor_Rate", 125.0)) * rng.uniform(0.92, 1.08)

        for _ in range(rooms_on_books):
            lead_time = max(1, int((stay_date - as_of_date).days + rng.randint(1, 90)))
            stay_nights = rng.choices([1, 2, 3, 4], weights=[0.45, 0.32, 0.17, 0.06])[0]
            rows.append(
                {
                    "booking_id": next_id,
                    "hotel": DEFAULT_HOTEL,
                    "arrival_date": stay_date,
                    "departure_date": stay_date + pd.Timedelta(days=stay_nights),
                    "booking_date": as_of_date - pd.Timedelta(days=rng.randint(0, min(lead_time, 120))),
                    "stay_nights": stay_nights,
                    "reservation_status": "Check-Out",
                    "reservation_status_date": stay_date + pd.Timedelta(days=stay_nights),
                    "is_canceled": 0,
                    "cancellation_date": pd.NaT,
                    "adr": round(adr * rng.uniform(0.9, 1.1), 2),
                    "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Corporate"]),
                    "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"]),
                    "customer_type": rng.choice(["Transient", "Transient-Party", "Contract"]),
                    "reserved_room_type": rng.choice(["A", "D", "E"]),
                    "assigned_room_type": rng.choice(["A", "D", "E"]),
                    "lead_time": lead_time,
                    "adults": rng.choice([1, 2, 2, 2, 3]),
                    "children": rng.choice([0, 0, 0, 1]),
                    "babies": 0,
                    "total_of_special_requests": rng.randint(0, 3),
                    "data_quality_flag": "synthetic_initial_otb",
                }
            )
            next_id += 1
    return pd.DataFrame(rows)


def initialize_live_ledger(output_file=LIVE_DATA_PATH, raw_path=RAW_BOOKINGS_PATH, as_of_date=DATA_END_DATE, seed=42):
    """Seed the live ledger with PMS-like bookings known by the demo as-of date."""
    bookings = normalize_bookings(raw_path, hotel=DEFAULT_HOTEL)
    as_of_date = pd.to_datetime(as_of_date)
    historical_seed = bookings[bookings["booking_date"].le(as_of_date)].copy()
    future_seed = _synthetic_future_on_books(as_of_date=as_of_date, seed=seed)
    ledger = pd.concat([historical_seed, future_seed], ignore_index=True)
    ledger.to_csv(output_file, index=False)
    return ledger


def _next_booking_id(df: pd.DataFrame) -> int:
    if df.empty or "booking_id" not in df.columns:
        return 1
    return int(pd.to_numeric(df["booking_id"], errors="coerce").max()) + 1


def simulate_live_booking_event(live_bookings_path=LIVE_DATA_PATH, as_of_date=DATA_END_DATE, seed=None):
    """Append a realistic synthetic PMS event for demo refreshes."""
    rng = random.Random(seed)
    if not os.path.exists(live_bookings_path):
        ledger = initialize_live_ledger(live_bookings_path, as_of_date=as_of_date)
    else:
        ledger = normalize_bookings(live_bookings_path, hotel=DEFAULT_HOTEL)

    as_of_date = pd.to_datetime(as_of_date)
    event_type = rng.choices(["new_booking", "cancellation", "adr_revision"], weights=[0.72, 0.18, 0.10])[0]

    if event_type == "new_booking" or ledger.empty:
        lead_time = rng.randint(0, 30)
        stay_nights = rng.choices([1, 2, 3, 4, 5], weights=[0.25, 0.32, 0.25, 0.12, 0.06])[0]
        arrival_date = as_of_date + pd.Timedelta(days=lead_time + 1)
        base_adr = rng.uniform(85, 180)
        weekend_lift = 1.12 if arrival_date.dayofweek in [4, 5] else 1.0
        booking = {
            "booking_id": _next_booking_id(ledger),
            "hotel": DEFAULT_HOTEL,
            "arrival_date": arrival_date,
            "departure_date": arrival_date + pd.Timedelta(days=stay_nights),
            "booking_date": as_of_date,
            "stay_nights": stay_nights,
            "reservation_status": "Check-Out",
            "reservation_status_date": arrival_date + pd.Timedelta(days=stay_nights),
            "is_canceled": 0,
            "cancellation_date": pd.NaT,
            "adr": round(base_adr * weekend_lift, 2),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Corporate"]),
            "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"]),
            "customer_type": rng.choice(["Transient", "Transient-Party", "Contract"]),
            "reserved_room_type": rng.choice(["A", "D", "E"]),
            "assigned_room_type": rng.choice(["A", "D", "E"]),
            "lead_time": lead_time,
            "adults": rng.choice([1, 2, 2, 2, 3]),
            "children": rng.choice([0, 0, 0, 1]),
            "babies": 0,
            "total_of_special_requests": rng.randint(0, 3),
            "data_quality_flag": "synthetic",
        }
        ledger = pd.concat([ledger, pd.DataFrame([booking])], ignore_index=True)
        summary = f"New booking for {arrival_date.date()} ({stay_nights} nights, ADR ${booking['adr']})"

    elif event_type == "cancellation":
        candidates = ledger[(ledger["is_canceled"].eq(0)) & (ledger["arrival_date"].gt(as_of_date))].copy()
        if candidates.empty:
            return simulate_live_booking_event(live_bookings_path, as_of_date=as_of_date, seed=rng.randint(0, 10_000))
        idx = rng.choice(list(candidates.index))
        ledger.loc[idx, "is_canceled"] = 1
        ledger.loc[idx, "reservation_status"] = "Canceled"
        ledger.loc[idx, "reservation_status_date"] = as_of_date
        ledger.loc[idx, "cancellation_date"] = as_of_date
        summary = f"Cancellation for booking {ledger.loc[idx, 'booking_id']}"

    else:
        candidates = ledger[(ledger["is_canceled"].eq(0)) & (ledger["arrival_date"].gt(as_of_date))].copy()
        if candidates.empty:
            return simulate_live_booking_event(live_bookings_path, as_of_date=as_of_date, seed=rng.randint(0, 10_000))
        idx = rng.choice(list(candidates.index))
        old_adr = float(ledger.loc[idx, "adr"])
        ledger.loc[idx, "adr"] = round(old_adr * rng.uniform(0.95, 1.12), 2)
        summary = f"ADR revision for booking {ledger.loc[idx, 'booking_id']}: ${old_adr:.2f} -> ${ledger.loc[idx, 'adr']:.2f}"

    ledger.to_csv(live_bookings_path, index=False)
    refresh_live_market_artifacts(ledger, as_of_date=as_of_date)
    return {"event_type": event_type, "summary": summary, "ledger_rows": len(ledger)}


def simulate_market(interval_seconds=5, as_of_date=DATA_END_DATE):
    print("Synthetic PMS simulator active. Press Ctrl+C to stop.")
    while True:
        event = simulate_live_booking_event(as_of_date=as_of_date)
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {event['summary']}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append synthetic PMS reservation events for the Hotel RMS demo.")
    parser.add_argument("--init", action="store_true", help="Initialize the live ledger with synthetic future OTB and exit.")
    parser.add_argument("--once", action="store_true", help="Append one event and exit.")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between synthetic events.")
    args = parser.parse_args()

    if args.init:
        ledger = initialize_live_ledger()
        refresh_live_market_artifacts(ledger)
        print(f"Initialized {LIVE_DATA_PATH} with {len(ledger)} PMS-like rows.")
    elif args.once:
        print(simulate_live_booking_event())
    else:
        simulate_market(interval_seconds=args.interval)
