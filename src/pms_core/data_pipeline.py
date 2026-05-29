import os
from typing import Iterable

import numpy as np
import pandas as pd

from project_core.config import BASE_CAPACITY, DATA_END_DATE, DEFAULT_HOTEL, RAW_BOOKINGS_PATH


MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

KEEP_COLUMNS = [
    "booking_id",
    "hotel",
    "arrival_date",
    "departure_date",
    "booking_date",
    "stay_nights",
    "reservation_status",
    "reservation_status_date",
    "is_canceled",
    "cancellation_date",
    "adr",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "reserved_room_type",
    "assigned_room_type",
    "lead_time",
    "adults",
    "children",
    "babies",
    "total_of_special_requests",
    "data_quality_flag",
]


def _parse_reservation_status_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def normalize_bookings(raw_path=RAW_BOOKINGS_PATH, hotel=DEFAULT_HOTEL) -> pd.DataFrame:
    """Normalize the Kaggle-style hotel bookings file into a PMS-like ledger."""
    df = pd.read_csv(raw_path)
    if hotel and "hotel" in df.columns:
        df = df[df["hotel"] == hotel].copy()

    if "booking_id" not in df.columns:
        df.insert(0, "booking_id", np.arange(1, len(df) + 1))

    if "arrival_date" not in df.columns:
        month_num = df["arrival_date_month"].map(MONTH_MAP)
        df["arrival_date"] = pd.to_datetime(
            {
                "year": df["arrival_date_year"],
                "month": month_num,
                "day": df["arrival_date_day_of_month"],
            },
            errors="coerce",
        )
    else:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")

    df["lead_time"] = pd.to_numeric(df.get("lead_time", 0), errors="coerce").fillna(0).clip(lower=0)
    if "stay_nights" in df.columns:
        df["stay_nights"] = pd.to_numeric(df["stay_nights"], errors="coerce").fillna(0).astype(int)
    else:
        df["stay_nights"] = (
            pd.to_numeric(df.get("stays_in_weekend_nights", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("stays_in_week_nights", 0), errors="coerce").fillna(0)
        ).astype(int)
    if "departure_date" in df.columns:
        df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")
    else:
        df["departure_date"] = df["arrival_date"] + pd.to_timedelta(df["stay_nights"], unit="D")
    if "booking_date" in df.columns:
        df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    else:
        df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = _parse_reservation_status_date(df["reservation_status_date"])
    else:
        df["reservation_status_date"] = df["booking_date"]

    df["reservation_status"] = df.get("reservation_status", "Check-Out").fillna("Check-Out")
    df["is_canceled"] = pd.to_numeric(df.get("is_canceled", 0), errors="coerce").fillna(0).astype(int)
    cancel_mask = df["is_canceled"].eq(1) | df["reservation_status"].isin(["Canceled", "No-Show"])
    df["cancellation_date"] = pd.NaT
    df.loc[cancel_mask, "cancellation_date"] = df.loc[cancel_mask, "reservation_status_date"]

    df["adr"] = pd.to_numeric(df.get("adr", 0), errors="coerce")
    invalid_adr = df["adr"].isna() | df["adr"].lt(0) | df["adr"].gt(1000)
    invalid_dates = df["arrival_date"].isna() | df["departure_date"].isna() | df["booking_date"].isna()
    invalid_los = df["stay_nights"].lt(1)

    df["data_quality_flag"] = "ok"
    df.loc[invalid_adr, "data_quality_flag"] = "invalid_adr"
    df.loc[invalid_los, "data_quality_flag"] = "invalid_los"
    df.loc[invalid_dates, "data_quality_flag"] = "invalid_date"
    df["adr"] = df["adr"].clip(lower=0, upper=1000)

    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[KEEP_COLUMNS].sort_values(["arrival_date", "booking_date", "booking_id"]).reset_index(drop=True)


def unroll_room_nights(bookings_df: pd.DataFrame, include_canceled: bool = False) -> pd.DataFrame:
    """Expand reservations into one row per occupied stay date."""
    df = bookings_df.copy()
    if not include_canceled:
        df = df[df["is_canceled"].eq(0)].copy()
    df = df[(df["stay_nights"] > 0) & df["arrival_date"].notna() & df["departure_date"].notna()]

    rows = []
    for row in df.itertuples(index=False):
        for stay_date in pd.date_range(row.arrival_date, row.departure_date - pd.Timedelta(days=1), freq="D"):
            rows.append(
                {
                    "booking_id": row.booking_id,
                    "stay_date": stay_date,
                    "adr": row.adr,
                    "room_revenue": row.adr,
                    "market_segment": row.market_segment,
                    "customer_type": row.customer_type,
                }
            )
    return pd.DataFrame(rows)


def _future_competitor_rates(dates: Iterable[pd.Timestamp], historical_daily: pd.DataFrame) -> pd.Series:
    rates = historical_daily[["Date", "ADR"]].dropna().copy()
    if rates.empty:
        return pd.Series([120.0] * len(list(dates)))

    rates["dow"] = rates["Date"].dt.dayofweek
    dow_avg = rates.groupby("dow")["ADR"].mean()
    overall = rates["ADR"].mean()
    values = []
    for date in dates:
        seasonal_rate = dow_avg.get(date.dayofweek, overall)
        values.append(round(float(seasonal_rate * 1.03), 2))
    return pd.Series(values)


def build_daily_hotel_data(
    raw_path=RAW_BOOKINGS_PATH,
    hotel=DEFAULT_HOTEL,
    capacity=BASE_CAPACITY,
    as_of_date=DATA_END_DATE,
    horizon_days=30,
) -> pd.DataFrame:
    """Build daily actuals plus a future plan horizon from the raw PMS-like ledger."""
    as_of_date = pd.to_datetime(as_of_date)
    bookings = normalize_bookings(raw_path, hotel=hotel)
    room_nights = unroll_room_nights(bookings)

    if room_nights.empty:
        raise ValueError("No valid room-night rows could be generated from the booking data.")

    actuals = (
        room_nights.groupby("stay_date")
        .agg(Occupied_Rooms=("booking_id", "count"), ADR=("adr", "mean"), Revenue=("room_revenue", "sum"))
        .reset_index()
        .rename(columns={"stay_date": "Date"})
    )

    cancellations = (
        bookings[bookings["cancellation_date"].notna()]
        .groupby("cancellation_date")
        .size()
        .rename("Cancellations")
        .reset_index()
        .rename(columns={"cancellation_date": "Date"})
    )
    pickup = (
        bookings.groupby("booking_date")
        .size()
        .rename("Bookings_Created")
        .reset_index()
        .rename(columns={"booking_date": "Date"})
    )

    date_index = pd.date_range(actuals["Date"].min(), as_of_date + pd.Timedelta(days=horizon_days), freq="D")
    daily = pd.DataFrame({"Date": date_index})
    daily = daily.merge(actuals, on="Date", how="left")
    daily = daily.merge(cancellations, on="Date", how="left")
    daily = daily.merge(pickup, on="Date", how="left")

    daily["Capacity"] = capacity
    daily["Occupied_Rooms"] = daily["Occupied_Rooms"].fillna(0)
    daily["Occupancy_Rate"] = daily["Occupied_Rooms"] / capacity
    daily.loc[daily["Date"].gt(as_of_date), ["Occupied_Rooms", "Occupancy_Rate", "ADR", "Revenue"]] = np.nan
    daily["RevPAR"] = daily["Revenue"] / capacity
    daily["Is_Weekend"] = daily["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    daily["Local_Event"] = 0
    daily["Cancellations"] = daily["Cancellations"].fillna(0).astype(int)
    daily["Bookings_Created"] = daily["Bookings_Created"].fillna(0).astype(int)
    daily["Booking_Pace"] = daily["Bookings_Created"] / capacity

    historical = daily[daily["Date"].le(as_of_date)].copy()
    daily["Competitor_Rate"] = daily["ADR"] * 1.03
    future_mask = daily["Date"].gt(as_of_date)
    daily.loc[future_mask, "Competitor_Rate"] = _future_competitor_rates(daily.loc[future_mask, "Date"], historical).values
    daily["Competitor_Rate"] = daily["Competitor_Rate"].ffill().bfill().round(2)

    columns = [
        "Date",
        "Occupancy_Rate",
        "Occupied_Rooms",
        "ADR",
        "Revenue",
        "RevPAR",
        "Is_Weekend",
        "Local_Event",
        "Competitor_Rate",
        "Capacity",
        "Bookings_Created",
        "Cancellations",
        "Booking_Pace",
    ]
    return daily[columns]


def refresh_daily_hotel_data(output_path, **kwargs) -> pd.DataFrame:
    daily = build_daily_hotel_data(**kwargs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily.to_csv(output_path, index=False)
    return daily
