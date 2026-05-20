from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MIN_SUPPORT = 30
DEFAULT_SMOOTHING = 10.0

RISK_HIERARCHY = [
    ("remaining_days_band", "lead_time_band", "market_segment", "distribution_channel", "customer_type"),
    ("remaining_days_band", "lead_time_band", "market_segment", "distribution_channel"),
    ("remaining_days_band", "lead_time_band", "market_segment"),
    ("remaining_days_band", "lead_time_band"),
    ("remaining_days_band",),
]

REMAINING_DAYS_SNAPSHOT = {
    "0-1": 1,
    "2-3": 3,
    "4-7": 7,
    "8-14": 14,
    "15-30": 30,
    "31+": 31,
}


def add_lead_time_band(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an interpretable lead-time bucket."""
    result = df.copy()
    lead_time = pd.to_numeric(result.get("lead_time", 0), errors="coerce").fillna(0).clip(lower=0)
    result["lead_time_band"] = pd.cut(
        lead_time,
        bins=[-np.inf, 7, 30, 90, np.inf],
        labels=["0-7", "8-30", "31-90", "91+"],
    ).astype("object")
    result["lead_time_band"] = result["lead_time_band"].fillna("Unknown")
    return result


def add_remaining_days_band(df: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Return a copy with current days-to-arrival and an interpretable risk band."""
    result = df.copy()
    as_of_date = pd.to_datetime(as_of_date)
    arrival_date = pd.to_datetime(result.get("arrival_date"), errors="coerce")
    days_to_arrival = (arrival_date - as_of_date).dt.days
    result["days_to_arrival"] = days_to_arrival

    result["remaining_days_band"] = np.select(
        [
            arrival_date.le(as_of_date),
            days_to_arrival.ge(0) & days_to_arrival.le(1),
            days_to_arrival.ge(2) & days_to_arrival.le(3),
            days_to_arrival.ge(4) & days_to_arrival.le(7),
            days_to_arrival.ge(8) & days_to_arrival.le(14),
            days_to_arrival.ge(15) & days_to_arrival.le(30),
            days_to_arrival.ge(31),
        ],
        ["arrived", "0-1", "2-3", "4-7", "8-14", "15-30", "31+"],
        default="Unknown",
    )
    return result


def _normalize_group_values(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for key in keys:
        if key not in result.columns:
            result[key] = "Unknown"
        result[key] = result[key].fillna("Unknown").astype(str)
    return result


def _historical_training_rows(bookings_df: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Build survival cohorts whose remaining cancellation outcome is known."""
    history = bookings_df.copy()
    as_of_date = pd.to_datetime(as_of_date)
    history["arrival_date"] = pd.to_datetime(history["arrival_date"], errors="coerce")
    history["booking_date"] = pd.to_datetime(history["booking_date"], errors="coerce")
    history["cancellation_date"] = pd.to_datetime(history.get("cancellation_date"), errors="coerce")
    history = history[
        history["arrival_date"].notna()
        & history["booking_date"].notna()
        & history["arrival_date"].le(as_of_date)
        & history["booking_date"].le(as_of_date)
    ].copy()
    if history.empty:
        return history

    cohorts = []
    for remaining_days_band, snapshot_days in REMAINING_DAYS_SNAPSHOT.items():
        snapshot_date = history["arrival_date"] - pd.to_timedelta(snapshot_days, unit="D")
        active_at_snapshot = history["booking_date"].le(snapshot_date) & (
            history["cancellation_date"].isna() | history["cancellation_date"].gt(snapshot_date)
        )
        cohort = history[active_at_snapshot].copy()
        if cohort.empty:
            continue
        cohort["days_to_arrival"] = snapshot_days
        cohort["remaining_days_band"] = remaining_days_band
        cohort["future_cancellation"] = (
            cohort["cancellation_date"].notna()
            & cohort["cancellation_date"].gt(snapshot_date.loc[cohort.index])
            & cohort["cancellation_date"].le(cohort["arrival_date"])
        ).astype(float)
        cohorts.append(cohort)

    if not cohorts:
        return pd.DataFrame(columns=list(history.columns) + ["days_to_arrival", "remaining_days_band", "future_cancellation"])

    return add_lead_time_band(pd.concat(cohorts, ignore_index=True))


def estimate_cancellation_probabilities(
    active_bookings: pd.DataFrame,
    historical_bookings: pd.DataFrame,
    as_of_date,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    smoothing: float = DEFAULT_SMOOTHING,
) -> pd.DataFrame:
    """Estimate interpretable booking-level cancellation probabilities.

    Rates are smoothed toward the global historical cancellation rate, then
    assigned from the most specific supported bucket down to the global rate.
    """
    if active_bookings.empty:
        return pd.DataFrame(
            {
                "cancellation_probability": pd.Series(dtype=float),
                "risk_source": pd.Series(dtype="object"),
            },
            index=active_bookings.index,
        )

    active = add_remaining_days_band(add_lead_time_band(active_bookings), as_of_date)
    history = _historical_training_rows(historical_bookings, as_of_date)
    global_rate = float(history["future_cancellation"].mean()) if not history.empty else 0.0
    band_rates = (
        history.groupby("remaining_days_band", dropna=False)["future_cancellation"].mean()
        if not history.empty
        else pd.Series(dtype=float)
    )

    probabilities = pd.Series(np.nan, index=active.index, dtype=float)
    sources = pd.Series("", index=active.index, dtype="object")
    arrived_mask = active["remaining_days_band"].eq("arrived")
    probabilities.loc[arrived_mask] = 0.0
    sources.loc[arrived_mask] = "arrived"

    for keys in RISK_HIERARCHY:
        if history.empty:
            break
        history_level = _normalize_group_values(history, keys)
        active_level = _normalize_group_values(active, keys)
        stats = (
            history_level.groupby(list(keys), dropna=False)["future_cancellation"]
            .agg(["count", "sum"])
            .reset_index()
        )
        stats = stats[stats["count"].ge(min_support)].copy()
        if stats.empty:
            continue
        stats["rate"] = (stats["sum"] + (smoothing * global_rate)) / (stats["count"] + smoothing)

        merged = (
            active_level.reset_index()
            .merge(stats[list(keys) + ["rate"]], on=list(keys), how="left")
            .set_index("index")
        )
        fill_mask = probabilities.isna() & merged["rate"].notna()
        probabilities.loc[fill_mask] = merged.loc[fill_mask, "rate"]
        sources.loc[fill_mask] = " + ".join(keys)

    band_fallback = active["remaining_days_band"].map(band_rates)
    fill_band_mask = probabilities.isna() & band_fallback.notna()
    probabilities.loc[fill_band_mask] = band_fallback.loc[fill_band_mask]
    sources.loc[fill_band_mask] = "remaining_days_band_global"

    probabilities = probabilities.fillna(global_rate).clip(lower=0.0, upper=1.0)
    sources = sources.mask(sources.eq(""), "global")
    return pd.DataFrame(
        {
            "days_to_arrival": active["days_to_arrival"],
            "remaining_days_band": active["remaining_days_band"],
            "cancellation_probability": probabilities,
            "risk_source": sources,
        },
        index=active.index,
    )
