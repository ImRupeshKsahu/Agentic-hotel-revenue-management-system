import os
from typing import Optional

import numpy as np
import pandas as pd

from cancellation_risk import estimate_cancellation_probabilities
from config import BASE_CAPACITY, DATA_END_DATE, DEFAULT_HOTEL, RAW_BOOKINGS_PATH
from data_pipeline import normalize_bookings


def _active_on_books(
    bookings_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    stay_dates: pd.Series,
    *,
    include_cancellation_risk: bool = False,
) -> pd.DataFrame:
    exploded = []
    stay_date_set = set(pd.to_datetime(stay_dates))
    candidate = bookings_df[
        (bookings_df["booking_date"].le(as_of_date))
        & (bookings_df["stay_nights"].gt(0))
        & (bookings_df["arrival_date"].notna())
        & (bookings_df["departure_date"].notna())
    ].copy()

    if include_cancellation_risk and not candidate.empty:
        risk = estimate_cancellation_probabilities(candidate, bookings_df, as_of_date)
        candidate = candidate.join(risk)
    else:
        candidate["cancellation_probability"] = 0.0

    for row in candidate.itertuples(index=False):
        cancel_date = getattr(row, "cancellation_date", pd.NaT)
        canceled_before_as_of = pd.notna(cancel_date) and cancel_date <= as_of_date
        if canceled_before_as_of:
            continue
        for stay_date in pd.date_range(row.arrival_date, row.departure_date - pd.Timedelta(days=1), freq="D"):
            if stay_date in stay_date_set:
                cancel_prob = float(getattr(row, "cancellation_probability", 0.0) or 0.0)
                cancel_prob = min(max(cancel_prob, 0.0), 1.0)
                exploded.append(
                    {
                        "stay_date": stay_date,
                        "booking_id": row.booking_id,
                        "adr": row.adr,
                        "cancellation_probability": cancel_prob,
                        "expected_retained_rooms": 1.0 - cancel_prob,
                        "expected_cancellations": cancel_prob,
                    }
                )
    return pd.DataFrame(exploded)


def calculate_otb_snapshot(
    bookings_df: pd.DataFrame,
    as_of_date=DATA_END_DATE,
    horizon_days: int = 30,
    capacity: int = BASE_CAPACITY,
) -> pd.DataFrame:
    """Calculate on-the-books rooms from the normalized PMS-like booking ledger."""
    as_of_date = pd.to_datetime(as_of_date)
    stay_dates = pd.Series(pd.date_range(as_of_date + pd.Timedelta(days=1), periods=horizon_days, freq="D"))
    active = _active_on_books(bookings_df, as_of_date, stay_dates, include_cancellation_risk=True)

    if active.empty:
        otb = pd.DataFrame(
            {
                "Date": stay_dates,
                "Live_OTB": 0,
                "Adjusted_OTB": 0.0,
                "Expected_Cancellations": 0.0,
                "OTB_ADR": np.nan,
            }
        )
    else:
        otb = (
            active.groupby("stay_date")
            .agg(
                Live_OTB=("booking_id", "count"),
                Adjusted_OTB=("expected_retained_rooms", "sum"),
                Expected_Cancellations=("expected_cancellations", "sum"),
                OTB_ADR=("adr", "mean"),
            )
            .reset_index()
            .rename(columns={"stay_date": "Date"})
        )
        otb = pd.DataFrame({"Date": stay_dates}).merge(otb, on="Date", how="left")

    otb["Live_OTB"] = otb["Live_OTB"].fillna(0).astype(int).clip(upper=capacity)
    otb["Adjusted_OTB"] = otb["Adjusted_OTB"].fillna(0).clip(lower=0, upper=otb["Live_OTB"]).round(2)
    otb["Expected_Cancellations"] = (otb["Live_OTB"] - otb["Adjusted_OTB"]).clip(lower=0).round(2)
    otb["OTB_Occupancy"] = otb["Live_OTB"] / capacity
    otb["Adjusted_OTB_Occupancy"] = (otb["Adjusted_OTB"] / capacity).round(4)
    otb["OTB_ADR"] = otb["OTB_ADR"].round(2)
    otb["Capacity"] = capacity
    otb["As_Of_Date"] = as_of_date

    historical_as_of = as_of_date - pd.DateOffset(years=1)
    historical = _active_on_books(bookings_df, historical_as_of, stay_dates - pd.DateOffset(years=1))
    if historical.empty:
        otb["Historical_Avg_OTB"] = otb["Live_OTB"].clip(lower=1)
    else:
        historical_counts = historical.groupby("stay_date").size().reset_index(name="Historical_Avg_OTB")
        historical_counts["Date"] = historical_counts["stay_date"] + pd.DateOffset(years=1)
        otb = otb.merge(historical_counts[["Date", "Historical_Avg_OTB"]], on="Date", how="left")
        otb["Historical_Avg_OTB"] = otb["Historical_Avg_OTB"].fillna(otb["Live_OTB"].rolling(7, min_periods=1).mean())

    otb["Historical_Avg_OTB"] = otb["Historical_Avg_OTB"].fillna(1).clip(lower=1).round().astype(int)
    otb["Booking_Velocity"] = (otb["Live_OTB"] / otb["Historical_Avg_OTB"]).replace([np.inf, -np.inf], 0).round(2)
    return otb


def load_booking_ledger(path: Optional[str] = None, hotel: str = DEFAULT_HOTEL) -> pd.DataFrame:
    source = path or RAW_BOOKINGS_PATH
    return normalize_bookings(source, hotel=hotel)


def load_live_otb_snapshot(
    as_of_date=DATA_END_DATE,
    horizon_days: int = 30,
    capacity: int = BASE_CAPACITY,
    path: Optional[str] = None,
) -> pd.DataFrame:
    bookings = load_booking_ledger(path=path)
    return calculate_otb_snapshot(bookings, as_of_date=as_of_date, horizon_days=horizon_days, capacity=capacity)


def export_live_market_state(snapshot_df: pd.DataFrame, competitor_rates: Optional[pd.DataFrame] = None, output_path: str = None):
    df = snapshot_df.copy()
    if competitor_rates is not None:
        rates = competitor_rates[["Date", "Competitor_Rate"]].copy()
        rates["Date"] = pd.to_datetime(rates["Date"])
        df = df.merge(rates, on="Date", how="left")
    df["Competitor_Rate"] = df.get("Competitor_Rate", pd.Series(dtype=float)).fillna(df["OTB_ADR"]).fillna(120.0)

    state = {}
    for row in df.itertuples(index=False):
        status = "Normal"
        if row.Booking_Velocity >= 1.2:
            status = "Ahead of historical pace"
        elif row.Booking_Velocity <= 0.8:
            status = "Behind historical pace"
        state[pd.to_datetime(row.Date).strftime("%Y-%m-%d")] = {
            "current_otb": int(row.Live_OTB),
            "adjusted_otb": round(float(getattr(row, "Adjusted_OTB", row.Live_OTB)), 2),
            "expected_cancellations": round(float(getattr(row, "Expected_Cancellations", 0.0)), 2),
            "adjusted_otb_occupancy": round(float(getattr(row, "Adjusted_OTB_Occupancy", row.Live_OTB / row.Capacity)), 4),
            "historical_avg_otb": int(row.Historical_Avg_OTB),
            "competitor_price": round(float(row.Competitor_Rate), 2),
            "total_rooms": int(row.Capacity),
            "booking_velocity": float(row.Booking_Velocity),
            "status": status,
        }

    if output_path:
        import json

        with open(output_path, "w") as f:
            json.dump(state, f, indent=4)
    return state
