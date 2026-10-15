"""Primary detector: one-sided binomial z-test (ML PRD section 6).

Pure functions plus a small streak store. The store is an interface so the
worker can swap the in-memory version for Redis (TTL 10 min) without touching
the detection logic.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from ml.config import Config, DetectorConfig
from packages.common.severity import severity


@dataclass(frozen=True)
class WindowStat:
    p_hat: float
    z: float
    ratio: float
    n: int
    f: int
    p0: float
    passed: bool


def clamp_baseline(p0: float, cfg: DetectorConfig) -> float:
    """Floor and cap so a zero-failure baseline cannot give infinite z."""
    return min(max(p0, cfg.p0_floor), cfg.p0_cap)


def evaluate_window(n: int, f: int, p0: float, cfg: DetectorConfig) -> WindowStat | None:
    """Return None when the window is too small to judge (never impute)."""
    p0 = clamp_baseline(p0, cfg)
    if n < cfg.min_n or f < cfg.min_failures:
        return None
    p_hat = f / n
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    ratio = p_hat / p0
    return WindowStat(p_hat, z, ratio, n, f, p0,
                      passed=(z >= cfg.z_min and ratio >= cfg.ratio_min))


def evaluate_sustained(parts: list[tuple[int, int]], p0: float,
                       cfg: DetectorConfig) -> WindowStat | None:
    """evaluate_window on the pooled window, plus a sustained-evidence check.

    `parts` is a list of (n, f) for equal, NON-overlapping time slices of the window.
    The pooled window must pass first. Then at least cfg.confirm_min_parts slices must each
    be elevated (rate >= part_ratio_min * baseline, with >= part_min_failures failures).
    This rejects a failure cluster crammed into one slice while keeping real, ongoing
    incidents. Disabled when cfg.confirm_parts <= 1.
    """
    n = sum(p[0] for p in parts)
    f = sum(p[1] for p in parts)
    stat = evaluate_window(n, f, p0, cfg)
    if stat is None or not stat.passed or cfg.confirm_parts <= 1:
        return stat
    p0c = clamp_baseline(p0, cfg)
    elevated = sum(1 for ni, fi in parts
                   if ni > 0 and fi >= cfg.part_min_failures and fi / ni >= cfg.part_ratio_min * p0c)
    return replace(stat, passed=elevated >= cfg.confirm_min_parts)


class StreakStore(Protocol):
    def get(self, key: str) -> int: ...
    def set(self, key: str, value: int) -> None: ...


class InMemoryStreakStore:
    """Test/dev store. Production uses Redis with TTL so a restart keeps streaks (M-06)."""

    def __init__(self) -> None:
        self._d: dict[str, int] = {}

    def get(self, key: str) -> int:
        return self._d.get(key, 0)

    def set(self, key: str, value: int) -> None:
        self._d[key] = value


def update_streak(segment_key: str, stat: WindowStat | None, store: StreakStore,
                  cfg: DetectorConfig) -> bool:
    """Advance the streak for a segment/metric. True when the signal should fire.

    A window that is too small (stat is None) or not anomalous resets the streak.
    The segment keeps firing on every tick while it stays anomalous; the incident
    engine de-duplicates by segment key.
    """
    streak = store.get(segment_key) + 1 if (stat and stat.passed) else 0
    store.set(segment_key, streak)
    return streak >= cfg.consecutive


def build_signal(*, stat: WindowStat, metric: str, segment: dict, window_start: datetime,
                 window_end: datetime, detected_at: datetime, amount_affected_ngn: float,
                 complaints_last_15min: int, top_failure_code: str | None,
                 cfg: Config, isolation_score: float | None = None) -> dict:
    """Build an AnomalySignal dict that validates against the ML PRD Listing 3 schema."""
    sev = severity(stat.f, stat.ratio, amount_affected_ngn, complaints_last_15min, cfg.severity)

    def iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "schema_version": "1.0",
        "signal_id": f"sig-{uuid.uuid4().hex[:8]}",
        "detected_at": iso(detected_at),
        "window": {"start": iso(window_start), "end": iso(window_end)},
        "segment": segment,
        "metric": metric,
        "detector": "binomial_z",
        "baseline": stat.p0,
        "observed": round(stat.p_hat, 4),
        "z_score": round(stat.z, 1),
        "ratio": round(stat.ratio, 2),
        "n_txn": stat.n,
        "n_failed": stat.f,
        "isolation_score": isolation_score,
        "top_failure_code": top_failure_code,
        "severity_score": round(sev.score, 2),
        "severity": sev.band,
    }
