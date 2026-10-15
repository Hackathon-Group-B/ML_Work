"""Batch scan: slide a window across a transaction table and apply the detector.

Used by demo.py and the multi-seed sensitivity sweep, so both run identical logic.
This is a simplified offline scan, not the live 30-second pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import DetectorConfig
from ml.detector.binomial import evaluate_sustained

# Scan geometry, shared by demo.py and sweep.py. 45-minute window cut into 3 slices of
# 15 minutes, advanced every 10 minutes. Chosen over 75 minutes because the 3-slice
# sustained rule needs the incident to fill every slice, so a shorter window catches
# shorter incidents (see the duration table printed by `python sweep.py`).
WINDOW_MINUTES = 45
STEP_MINUTES = 10
PARTS = 3
WINDOW = pd.Timedelta(minutes=WINDOW_MINUTES)
STEP = pd.Timedelta(minutes=STEP_MINUTES)


def window_counts(df: pd.DataFrame, route: str, t_start: pd.Timestamp, t_end: pd.Timestamp,
                  window: pd.Timedelta, step: pd.Timedelta, parts: int):
    """Per-slice (n, f) counts for every window position on one route.

    Returns (window_starts, counts) where counts has shape (n_windows, parts, 2).
    Computed once per dataset so different rules can be compared cheaply.
    """
    r = df[df.route == route].sort_values("timestamp")
    ts = r.timestamp.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    failed = (r.outcome == "FAILED").to_numpy().astype(int)
    cum = np.concatenate([[0], np.cumsum(failed)])

    starts = []
    t = t_start
    while t + window <= t_end:
        starts.append(t)
        t += step
    slice_len = window / parts
    counts = np.zeros((len(starts), parts, 2), dtype=int)
    for i, s in enumerate(starts):
        edges = [np.datetime64((s + slice_len * k).tz_convert("UTC").tz_localize(None))
                 for k in range(parts + 1)]
        idx = np.searchsorted(ts, edges, side="left")
        for k in range(parts):
            counts[i, k, 0] = idx[k + 1] - idx[k]
            counts[i, k, 1] = cum[idx[k + 1]] - cum[idx[k]]
    return starts, counts


def fire_windows(starts, counts, p0: float, cfg: DetectorConfig, window: pd.Timedelta):
    """Apply the detector to precomputed counts. Returns list of fired windows."""
    fired = []
    for s, c in zip(starts, counts):
        stat = evaluate_sustained([(int(a), int(b)) for a, b in c], p0, cfg)
        if stat is not None and stat.passed:
            fired.append((s, s + window, stat.n, stat.f, float(stat.z), float(stat.ratio)))
    return fired


def merge_into_episodes(fired):
    """Overlapping fired windows are one incident seen several times: merge them."""
    episodes = []
    for start, end, n, f, z, ratio in sorted(fired):
        if episodes and start <= episodes[-1]["end"]:
            ep = episodes[-1]
            ep["end"] = max(ep["end"], end)
            ep["windows"] += 1
            if z > ep["peak_z"]:
                ep["peak_z"], ep["peak_ratio"] = z, ratio
        else:
            episodes.append({"start": start, "end": end, "windows": 1,
                             "peak_z": z, "peak_ratio": ratio})
    return episodes
