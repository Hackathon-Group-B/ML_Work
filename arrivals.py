"""Non-homogeneous Poisson arrival process (ML PRD section 9, Table 3).

Hour-of-day profile with a midday peak, a weekday multiplier, and a month-end
uplift, applied independently per channel (we treat "channel" as txn_type here,
since the generator does not yet model separate delivery channels).
"""
from __future__ import annotations

import numpy as np

from ml.config import GeneratorConfig

# Relative intensity by hour of day (0..23). Midday peak, quiet overnight.
HOUR_OF_DAY_PROFILE = np.array([
    0.25, 0.15, 0.10, 0.10, 0.15, 0.30, 0.55, 0.85,   # 00-07
    1.10, 1.30, 1.45, 1.55, 1.60, 1.50, 1.40, 1.35,   # 08-15
    1.30, 1.25, 1.15, 1.00, 0.80, 0.60, 0.45, 0.35,   # 16-23
])

# Monday=0 .. Sunday=6. Weekdays busier than weekends.
WEEKDAY_MULTIPLIER = np.array([1.05, 1.05, 1.05, 1.05, 1.10, 0.80, 0.70])

MONTH_END_UPLIFT = 1.25   # applied to the last 2 days of the month


def hourly_intensity(timestamps_hour: np.ndarray, weekday: np.ndarray, is_month_end: np.ndarray) -> np.ndarray:
    """Relative arrival-rate multiplier for each hour slot (not yet normalised to a target volume)."""
    base = HOUR_OF_DAY_PROFILE[timestamps_hour] * WEEKDAY_MULTIPLIER[weekday]
    return base * np.where(is_month_end, MONTH_END_UPLIFT, 1.0)


def build_hour_buckets(start, days: int):
    """Return per-hour (timestamp, weekday, is_month_end) for `days` days starting at `start`."""
    import pandas as pd
    hours = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    weekday = hours.weekday.to_numpy()
    is_month_end = np.asarray(hours.is_month_end) | np.asarray((hours + pd.Timedelta(days=1)).is_month_end)
    return hours, hours.hour.to_numpy(), weekday, is_month_end


def allocate_daily_counts(cfg: GeneratorConfig, start, rng: np.random.Generator):
    """Split the target daily volume across every hour of the history using the intensity profile.

    Returns (hour_timestamps, counts_per_hour) with counts summing to
    cfg.target_txn_per_day * cfg.history_days (in expectation; Poisson adds noise).
    """
    hours, hod, weekday, month_end = build_hour_buckets(start, cfg.history_days)
    weights = hourly_intensity(hod, weekday, month_end)
    weights = weights / weights.sum()
    total_target = cfg.target_txn_per_day * cfg.history_days
    expected_per_hour = weights * total_target
    counts = rng.poisson(expected_per_hour)
    return hours, counts
