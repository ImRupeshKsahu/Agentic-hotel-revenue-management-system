from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MIN_SUPPORT = 30
DEFAULT_SMOOTHING = 10.0

RISK_HIERARCHY = [
    ("lead_time_band", "market_segment", "distribution_channel", "customer_type"),
    ("lead_time_band", "market_segment", "distribution_channel"),
    ("lead_time_band", "market_segment"),
    ("lead_time_band",),
]


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


def _normalize_group_values(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for key in keys:
        if key not in result.columns:
            result[key] = "Unknown"
        result[key] = result[key].fillna("Unknown").astype(str)
    return result


def _historical_training_rows(bookings_df: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Use only bookings whose stay outcome should be known by the as-of date."""
    history = bookings_df.copy()
    as_of_date = pd.to_datetime(as_of_date)
    history["arrival_date"] = pd.to_datetime(history["arrival_date"], errors="coerce")
    history["booking_date"] = pd.to_datetime(history["booking_date"], errors="coerce")
    history = history[
        history["arrival_date"].notna()
        & history["booking_date"].notna()
        & history["arrival_date"].le(as_of_date)
        & history["booking_date"].le(as_of_date)
    ].copy()
    history["is_canceled"] = pd.to_numeric(history.get("is_canceled", 0), errors="coerce").fillna(0).clip(0, 1)
    return add_lead_time_band(history)


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

    active = add_lead_time_band(active_bookings)
    history = _historical_training_rows(historical_bookings, as_of_date)
    global_rate = float(history["is_canceled"].mean()) if not history.empty else 0.0

    probabilities = pd.Series(np.nan, index=active.index, dtype=float)
    sources = pd.Series("", index=active.index, dtype="object")

    for keys in RISK_HIERARCHY:
        if history.empty:
            break
        history_level = _normalize_group_values(history, keys)
        active_level = _normalize_group_values(active, keys)
        stats = (
            history_level.groupby(list(keys), dropna=False)["is_canceled"]
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

    probabilities = probabilities.fillna(global_rate).clip(lower=0.0, upper=1.0)
    sources = sources.mask(sources.eq(""), "global")
    return pd.DataFrame(
        {
            "cancellation_probability": probabilities,
            "risk_source": sources,
        },
        index=active.index,
    )
