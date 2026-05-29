import os
import random
from typing import Iterable, Optional

import pandas as pd

from project_core.config import DATA_END_DATE, LIVE_COMPETITOR_MARKET_PATH


MARKET_SNAPSHOT_COLUMNS = [
    "stay_date",
    "as_of_timestamp",
    "comp_low",
    "comp_median",
    "comp_high",
    "sample_size",
    "source_quality",
    "market_regime",
]

REGIME_FACTORS = {
    "normal_market": (0.94, 1.00, 1.07),
    "event_compression": (1.02, 1.10, 1.18),
    "market_wide_sellout": (1.08, 1.18, 1.30),
    "aggressive_discounter": (0.82, 0.98, 1.08),
    "delayed_response": (0.92, 1.00, 1.05),
}

DEFAULT_REGIME_CYCLE = [
    "normal_market",
    "event_compression",
    "normal_market",
    "market_wide_sellout",
    "aggressive_discounter",
    "normal_market",
    "delayed_response",
]


def _normalize_dates(stay_dates: Iterable[pd.Timestamp]) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stay_dates))).dropna().dt.normalize()


def _baseline_by_date(
    stay_dates: pd.Series,
    baseline_rates: Optional[pd.DataFrame] = None,
) -> dict[pd.Timestamp, float]:
    if baseline_rates is None or baseline_rates.empty:
        return {date: 120.0 for date in stay_dates}

    rates = baseline_rates.copy()
    if "Date" in rates.columns:
        date_col = "Date"
    elif "stay_date" in rates.columns:
        date_col = "stay_date"
    else:
        return {date: 120.0 for date in stay_dates}

    value_col = "Competitor_Rate" if "Competitor_Rate" in rates.columns else "comp_median"
    if value_col not in rates.columns:
        return {date: 120.0 for date in stay_dates}

    rates[date_col] = pd.to_datetime(rates[date_col]).dt.normalize()
    rates[value_col] = pd.to_numeric(rates[value_col], errors="coerce")
    medians = rates.dropna(subset=[value_col]).set_index(date_col)[value_col].to_dict()
    fallback = float(rates[value_col].dropna().median()) if rates[value_col].notna().any() else 120.0
    return {date: float(medians.get(date, fallback)) for date in stay_dates}


def build_simulated_market_snapshot(
    stay_dates: Iterable[pd.Timestamp],
    *,
    as_of_timestamp=DATA_END_DATE,
    baseline_rates: Optional[pd.DataFrame] = None,
    seed: int = 42,
    regime_cycle: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create future-ready simulated market observations for each stay date."""
    dates = _normalize_dates(stay_dates)
    baselines = _baseline_by_date(dates, baseline_rates=baseline_rates)
    regimes = regime_cycle or DEFAULT_REGIME_CYCLE
    rng = random.Random(seed)
    rows = []
    as_of_timestamp = pd.Timestamp(as_of_timestamp).isoformat()

    for index, stay_date in enumerate(dates):
        regime = regimes[index % len(regimes)]
        low_factor, median_factor, high_factor = REGIME_FACTORS[regime]
        baseline = baselines[stay_date]
        jitter = rng.uniform(-0.015, 0.015)
        median = round(baseline * (median_factor + jitter), 2)
        low = round(min(median, baseline * (low_factor + jitter / 2)), 2)
        high = round(max(median, baseline * (high_factor + jitter)), 2)
        rows.append(
            {
                "stay_date": stay_date.strftime("%Y-%m-%d"),
                "as_of_timestamp": as_of_timestamp,
                "comp_low": low,
                "comp_median": median,
                "comp_high": high,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": regime,
            }
        )

    return pd.DataFrame(rows, columns=MARKET_SNAPSHOT_COLUMNS)


def save_market_snapshot(snapshot_df: pd.DataFrame, output_path: str = LIVE_COMPETITOR_MARKET_PATH) -> str:
    snapshot_df.to_csv(output_path, index=False)
    return output_path


def load_market_snapshot(path: str = LIVE_COMPETITOR_MARKET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=MARKET_SNAPSHOT_COLUMNS)
    snapshot = pd.read_csv(path)
    missing = [column for column in MARKET_SNAPSHOT_COLUMNS if column not in snapshot.columns]
    for column in missing:
        snapshot[column] = pd.NA
    return snapshot[MARKET_SNAPSHOT_COLUMNS]


def initialize_competitor_market(
    stay_dates: Iterable[pd.Timestamp],
    *,
    baseline_rates: Optional[pd.DataFrame] = None,
    as_of_timestamp=DATA_END_DATE,
    output_path: str = LIVE_COMPETITOR_MARKET_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    snapshot = build_simulated_market_snapshot(
        stay_dates,
        as_of_timestamp=as_of_timestamp,
        baseline_rates=baseline_rates,
        seed=seed,
    )
    save_market_snapshot(snapshot, output_path=output_path)
    return snapshot


def ensure_competitor_market(
    stay_dates: Iterable[pd.Timestamp],
    *,
    baseline_rates: Optional[pd.DataFrame] = None,
    as_of_timestamp=DATA_END_DATE,
    path: str = LIVE_COMPETITOR_MARKET_PATH,
) -> pd.DataFrame:
    existing = load_market_snapshot(path)
    requested_dates = {date.strftime("%Y-%m-%d") for date in _normalize_dates(stay_dates)}
    existing_dates = set(existing["stay_date"].dropna().astype(str))
    if existing.empty or not requested_dates.issubset(existing_dates):
        return initialize_competitor_market(
            stay_dates,
            baseline_rates=baseline_rates,
            as_of_timestamp=as_of_timestamp,
            output_path=path,
        )
    return existing


def simulate_competitor_market_event(
    *,
    stay_dates: Iterable[pd.Timestamp],
    baseline_rates: Optional[pd.DataFrame] = None,
    as_of_timestamp=DATA_END_DATE,
    path: str = LIVE_COMPETITOR_MARKET_PATH,
    seed: Optional[int] = None,
) -> dict:
    """Mutate one simulated stay-date observation to mimic a live market move."""
    snapshot = ensure_competitor_market(
        stay_dates,
        baseline_rates=baseline_rates,
        as_of_timestamp=as_of_timestamp,
        path=path,
    )
    rng = random.Random(seed)
    if snapshot.empty:
        raise ValueError("Cannot simulate a market event without stay dates.")

    row_index = rng.choice(list(snapshot.index))
    old_regime = snapshot.loc[row_index, "market_regime"]
    candidate_regimes = [regime for regime in REGIME_FACTORS if regime != old_regime]
    new_regime = rng.choice(candidate_regimes)
    target_date = pd.Timestamp(snapshot.loc[row_index, "stay_date"])

    one_row = build_simulated_market_snapshot(
        [target_date],
        as_of_timestamp=as_of_timestamp,
        baseline_rates=baseline_rates,
        seed=rng.randint(0, 10_000),
        regime_cycle=[new_regime],
    ).iloc[0]
    for column in MARKET_SNAPSHOT_COLUMNS:
        snapshot.loc[row_index, column] = one_row[column]

    save_market_snapshot(snapshot, output_path=path)
    return {
        "stay_date": target_date.strftime("%Y-%m-%d"),
        "old_regime": old_regime,
        "new_regime": new_regime,
        "comp_median": float(one_row["comp_median"]),
        "summary": (
            f"Market move for {target_date.date()}: {old_regime} -> {new_regime}; "
            f"comp median now ${float(one_row['comp_median']):.2f}"
        ),
    }
