import os
from typing import Optional

import numpy as np
import pandas as pd

from pricing_core.cancellation import estimate_cancellation_probabilities
from config import BASE_CAPACITY, DATA_END_DATE, DEFAULT_HOTEL, RAW_BOOKINGS_PATH
from pms_core.data_pipeline import normalize_bookings

PACE_SMOOTHING_ROOMS = 10.0
PICKUP_WINDOW_DAYS = 7


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
                        "is_stayover": bool(row.arrival_date <= as_of_date),
                        "cancellation_probability": cancel_prob,
                        "expected_retained_rooms": 1.0 - cancel_prob,
                        "expected_cancellations": cancel_prob,
                    }
                )
    return pd.DataFrame(exploded)


def _aggregate_active(active: pd.DataFrame, stay_dates: pd.Series) -> pd.DataFrame:
    if active.empty:
        agg = pd.DataFrame({"Date": stay_dates})
        agg["Gross_OTB"] = 0
        agg["Stayover_OTB"] = 0
        agg["Future_Arrival_OTB"] = 0
        agg["Adjusted_OTB"] = 0.0
        agg["Expected_Cancellations"] = 0.0
        agg["OTB_ADR"] = np.nan
        return agg

    agg = (
        active.groupby("stay_date")
        .agg(
            Gross_OTB=("booking_id", "count"),
            Stayover_OTB=("is_stayover", "sum"),
            Adjusted_OTB=("expected_retained_rooms", "sum"),
            Expected_Cancellations=("expected_cancellations", "sum"),
            OTB_ADR=("adr", "mean"),
        )
        .reset_index()
        .rename(columns={"stay_date": "Date"})
    )
    agg["Stayover_OTB"] = agg["Stayover_OTB"].astype(int)
    agg["Future_Arrival_OTB"] = agg["Gross_OTB"] - agg["Stayover_OTB"]
    return pd.DataFrame({"Date": stay_dates}).merge(agg, on="Date", how="left")


def _pace_ratio(current, historical):
    return (current + PACE_SMOOTHING_ROOMS) / (historical + PACE_SMOOTHING_ROOMS)


def _pickup_trend_index(current_pickup, historical_pickup):
    denominator = np.maximum(np.abs(historical_pickup), PACE_SMOOTHING_ROOMS)
    relative_change = (current_pickup - historical_pickup) / denominator
    return 1 + np.clip(relative_change, -0.50, 0.50)


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
    previous_active = _active_on_books(
        bookings_df,
        as_of_date - pd.Timedelta(days=PICKUP_WINDOW_DAYS),
        stay_dates,
        include_cancellation_risk=True,
    )

    otb = _aggregate_active(active, stay_dates)
    previous = _aggregate_active(previous_active, stay_dates)[["Date", "Gross_OTB", "Adjusted_OTB"]].rename(
        columns={
            "Gross_OTB": "Previous_Gross_OTB",
            "Adjusted_OTB": "Previous_Adjusted_OTB",
        }
    )
    otb = otb.merge(previous, on="Date", how="left")

    otb["Gross_OTB"] = otb["Gross_OTB"].fillna(0).astype(int)
    otb["Stayover_OTB"] = otb["Stayover_OTB"].fillna(0).astype(int)
    otb["Future_Arrival_OTB"] = otb["Future_Arrival_OTB"].fillna(0).astype(int)
    otb["Live_OTB"] = otb["Gross_OTB"].clip(upper=capacity)
    otb["Stayover_OTB"] = otb[["Stayover_OTB", "Live_OTB"]].min(axis=1).astype(int)
    otb["Future_Arrival_OTB"] = (otb["Live_OTB"] - otb["Stayover_OTB"]).clip(lower=0).astype(int)
    otb["Adjusted_OTB"] = otb["Adjusted_OTB"].fillna(0).clip(lower=0, upper=otb["Live_OTB"]).round(2)
    otb["Expected_Cancellations"] = (otb["Live_OTB"] - otb["Adjusted_OTB"]).clip(lower=0).round(2)
    otb["Previous_Gross_OTB"] = otb["Previous_Gross_OTB"].fillna(0).astype(int)
    otb["Previous_Adjusted_OTB"] = otb["Previous_Adjusted_OTB"].fillna(0).clip(lower=0).round(2)
    otb["Net_Pickup_7d"] = otb["Gross_OTB"] - otb["Previous_Gross_OTB"]
    otb["OTB_Occupancy"] = otb["Live_OTB"] / capacity
    otb["Adjusted_OTB_Occupancy"] = (otb["Adjusted_OTB"] / capacity).round(4)
    otb["OTB_ADR"] = otb["OTB_ADR"].round(2)
    otb["Capacity"] = capacity
    otb["As_Of_Date"] = as_of_date

    historical_as_of = as_of_date - pd.DateOffset(years=1)
    historical_stay_dates = stay_dates - pd.DateOffset(years=1)
    historical = _active_on_books(
        bookings_df,
        historical_as_of,
        historical_stay_dates,
        include_cancellation_risk=True,
    )
    historical_previous = _active_on_books(
        bookings_df,
        historical_as_of - pd.Timedelta(days=PICKUP_WINDOW_DAYS),
        historical_stay_dates,
        include_cancellation_risk=True,
    )
    if historical.empty:
        otb["Historical_Gross_OTB"] = np.nan
        otb["Historical_Adjusted_OTB"] = np.nan
    else:
        historical_current = _aggregate_active(historical, historical_stay_dates)[
            ["Date", "Gross_OTB", "Adjusted_OTB"]
        ].rename(
            columns={
                "Gross_OTB": "Historical_Gross_OTB",
                "Adjusted_OTB": "Historical_Adjusted_OTB",
            }
        )
        historical_current["Date"] = historical_current["Date"] + pd.DateOffset(years=1)
        otb = otb.merge(historical_current, on="Date", how="left")

    if historical_previous.empty:
        historical_prev = pd.DataFrame(
            {
                "Date": stay_dates,
                "Historical_Previous_Gross_OTB": np.nan,
            }
        )
    else:
        historical_prev = _aggregate_active(historical_previous, historical_stay_dates)[["Date", "Gross_OTB"]].rename(
            columns={"Gross_OTB": "Historical_Previous_Gross_OTB"}
        )
        historical_prev["Date"] = historical_prev["Date"] + pd.DateOffset(years=1)
    otb = otb.merge(historical_prev, on="Date", how="left")

    historical_columns = [
        "Historical_Gross_OTB",
        "Historical_Adjusted_OTB",
        "Historical_Previous_Gross_OTB",
    ]
    otb["Pace_Confidence"] = np.where(
        otb[historical_columns].notna().all(axis=1),
        "high",
        "low",
    )

    otb["Historical_Gross_OTB"] = otb["Historical_Gross_OTB"].fillna(otb["Gross_OTB"]).clip(lower=0)
    otb["Historical_Adjusted_OTB"] = (
        otb["Historical_Adjusted_OTB"].fillna(otb["Adjusted_OTB"]).clip(lower=0).round(2)
    )
    otb["Historical_Previous_Gross_OTB"] = (
        otb["Historical_Previous_Gross_OTB"].fillna(otb["Previous_Gross_OTB"]).clip(lower=0)
    )
    otb["Historical_Net_Pickup_7d"] = otb["Historical_Gross_OTB"] - otb["Historical_Previous_Gross_OTB"]

    otb["Gross_Pace_Index"] = _pace_ratio(otb["Gross_OTB"], otb["Historical_Gross_OTB"]).round(2)
    otb["Retained_Pace_Index"] = _pace_ratio(
        otb["Adjusted_OTB"],
        otb["Historical_Adjusted_OTB"],
    ).round(2)
    otb["Pickup_Trend_Index"] = _pickup_trend_index(
        otb["Net_Pickup_7d"],
        otb["Historical_Net_Pickup_7d"],
    ).round(2)
    otb["Pricing_Pace_Index"] = (
        (0.20 * otb["Gross_Pace_Index"])
        + (0.50 * otb["Retained_Pace_Index"])
        + (0.30 * otb["Pickup_Trend_Index"])
    ).clip(lower=0.75, upper=1.25).round(2)

    otb["Historical_Avg_OTB"] = otb["Historical_Gross_OTB"].clip(lower=1).round().astype(int)
    otb["Booking_Velocity"] = otb["Gross_Pace_Index"]
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


def export_live_market_state(
    snapshot_df: pd.DataFrame,
    competitor_rates: Optional[pd.DataFrame] = None,
    market_snapshots: Optional[pd.DataFrame] = None,
    output_path: str = None,
):
    df = snapshot_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if competitor_rates is not None:
        rates = competitor_rates[["Date", "Competitor_Rate"]].copy()
        rates["Date"] = pd.to_datetime(rates["Date"])
        df = df.merge(rates, on="Date", how="left")

    if market_snapshots is not None and not market_snapshots.empty:
        market = market_snapshots.copy()
        market["Date"] = pd.to_datetime(market["stay_date"])
        keep = [
            "Date",
            "as_of_timestamp",
            "comp_low",
            "comp_median",
            "comp_high",
            "sample_size",
            "source_quality",
            "market_regime",
        ]
        df = df.merge(market[keep], on="Date", how="left")

    if "Competitor_Rate" not in df.columns:
        df["Competitor_Rate"] = np.nan
    df["Competitor_Rate"] = df["Competitor_Rate"].fillna(df["OTB_ADR"]).fillna(120.0)
    for column in ["comp_median", "comp_low", "comp_high"]:
        if column not in df.columns:
            df[column] = np.nan
    df["comp_median"] = df["comp_median"].fillna(df["Competitor_Rate"])
    df["comp_low"] = df["comp_low"].fillna(df["comp_median"])
    df["comp_high"] = df["comp_high"].fillna(df["comp_median"])
    if "sample_size" not in df.columns:
        df["sample_size"] = np.nan
    df["sample_size"] = df["sample_size"].fillna(1).astype(int)
    if "source_quality" not in df.columns:
        df["source_quality"] = pd.NA
    if "market_regime" not in df.columns:
        df["market_regime"] = pd.NA
    if "as_of_timestamp" not in df.columns:
        df["as_of_timestamp"] = pd.NA
    df["source_quality"] = df["source_quality"].fillna("legacy_single_rate")
    df["market_regime"] = df["market_regime"].fillna("legacy_single_rate")
    df["as_of_timestamp"] = df["as_of_timestamp"].fillna(pd.Timestamp(DATA_END_DATE).isoformat())

    state = {}
    for row in df.itertuples(index=False):
        status = "Normal"
        gross_pace_index = float(getattr(row, "Gross_Pace_Index", row.Booking_Velocity))
        retained_pace_index = float(getattr(row, "Retained_Pace_Index", gross_pace_index))
        pickup_trend_index = float(getattr(row, "Pickup_Trend_Index", gross_pace_index))
        pricing_pace_index = float(getattr(row, "Pricing_Pace_Index", gross_pace_index))
        if gross_pace_index >= 1.2:
            status = "Ahead of historical pace"
        elif gross_pace_index <= 0.8:
            status = "Behind historical pace"
        state[pd.to_datetime(row.Date).strftime("%Y-%m-%d")] = {
            "current_otb": int(row.Live_OTB),
            "raw_otb_occupancy": round(float(row.OTB_Occupancy), 4),
            "stayover_otb": int(getattr(row, "Stayover_OTB", 0)),
            "future_arrival_otb": int(getattr(row, "Future_Arrival_OTB", row.Live_OTB)),
            "adjusted_otb": round(float(getattr(row, "Adjusted_OTB", row.Live_OTB)), 2),
            "expected_cancellations": round(float(getattr(row, "Expected_Cancellations", 0.0)), 2),
            "adjusted_otb_occupancy": round(float(getattr(row, "Adjusted_OTB_Occupancy", row.Live_OTB / row.Capacity)), 4),
            "historical_avg_otb": int(row.Historical_Avg_OTB),
            "booked_adr": round(float(row.OTB_ADR), 2),
            "competitor_price": round(float(row.comp_median), 2),
            "comp_low": round(float(row.comp_low), 2),
            "comp_median": round(float(row.comp_median), 2),
            "comp_high": round(float(row.comp_high), 2),
            "sample_size": int(row.sample_size),
            "source_quality": str(row.source_quality),
            "market_regime": str(row.market_regime),
            "market_as_of_timestamp": str(row.as_of_timestamp),
            "total_rooms": int(row.Capacity),
            "gross_otb": int(getattr(row, "Gross_OTB", row.Live_OTB)),
            "net_pickup_7d": int(getattr(row, "Net_Pickup_7d", 0)),
            "historical_net_pickup_7d": int(getattr(row, "Historical_Net_Pickup_7d", 0)),
            "gross_pace_index": gross_pace_index,
            "retained_pace_index": retained_pace_index,
            "pickup_trend_index": pickup_trend_index,
            "pricing_pace_index": pricing_pace_index,
            "pace_confidence": str(getattr(row, "Pace_Confidence", "low")),
            "booking_velocity": gross_pace_index,
            "status": status,
        }

    if output_path:
        import json

        with open(output_path, "w") as f:
            json.dump(state, f, indent=4)
    return state
