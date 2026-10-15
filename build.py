"""Main entry point: generate_dataset() ties population + arrivals + transactions +
scenarios + labels into one reproducible call (ML PRD section 9, M-02).

Usage:
    from ml.generator.build import generate_dataset
    df, labels = generate_dataset(seed=42)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import Config, load_config
from ml.generator.arrivals import allocate_daily_counts
from ml.generator.labels import write_labels
from ml.generator.population import build_population
from ml.generator.scenarios import inject_low_volume_burst, inject_spike
from ml.generator.transactions import generate_history

DEFAULT_START = pd.Timestamp("2026-10-13", tz="UTC")   # a Tuesday; 7 days -> next Tuesday


def generate_dataset(cfg: Config | None = None, *, seed: int | None = None,
                     start: pd.Timestamp = DEFAULT_START, inject_demo_scenarios: bool = True):
    """Generate baseline history, optionally inject demo scenarios 2 and 6, and return
    (transactions_df, labels_list). Deterministic: same seed + config -> same output.
    """
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg.generator.seed
    rng = np.random.default_rng(seed)

    population = build_population(cfg.generator, rng)
    hours, counts = allocate_daily_counts(cfg.generator, start, rng)
    df = generate_history(cfg, population, hours, counts, rng)

    labels = []
    if inject_demo_scenarios:
        # Scenario 2: spike on NIP_ROUTE_2 to 18.7% (the PRD's rate constant), during a
        # peak hour. Window widened to 75 minutes (not the PRD's illustrative 14 minutes) so
        # that it accumulates enough volume to clear the detector's min_n/min_failures gates
        # at our actual per-route throughput -- see the "Volume/worked-example mismatch" note
        # in config/defaults.yaml.
        spike_day = start + pd.Timedelta(days=3)
        spike_start = spike_day.replace(hour=14, minute=0, second=0, microsecond=0)
        df, spike_label = inject_spike(
            df, route="NIP_ROUTE_2", start=spike_start, duration_minutes=75,
            fail_probability=0.187, failure_code="E_BENEFICIARY_TIMEOUT",
            magnitude="MEDIUM", rng=rng)
        labels.append(spike_label)

        # Scenario 6 (hard negative): 12 failures at low volume, must not raise a signal.
        # Picked during a normal-traffic hour (not overnight) so there are enough candidate
        # transactions on the route to host the burst.
        burst_day = start + pd.Timedelta(days=1)
        burst_start = burst_day.replace(hour=10, minute=0, second=0, microsecond=0)
        df, burst_label = inject_low_volume_burst(
            df, route="INTRA_CORE", start=burst_start, duration_minutes=5,
            n_failures=12, rng=rng)
        labels.append(burst_label)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, labels


def generate_and_save(out_dir: str | Path, cfg: Config | None = None, seed: int | None = None):
    """Generate the dataset and write transactions.csv + labels.jsonl + refs files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, labels = generate_dataset(cfg, seed=seed)
    df.to_csv(out_dir / "transactions.csv", index=False)
    write_labels(labels, out_dir)
    return df, labels
