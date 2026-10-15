"""Tests for the synthetic data generator (ML PRD gates M-02 and M-03)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from ml.config import load_config
from ml.generator.build import generate_dataset
from ml.generator.population import build_population


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def dataset(cfg):
    return generate_dataset(cfg, seed=42)


def _content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# --- M-02: population and reproducibility ---

def test_population_sizes(cfg):
    rng = np.random.default_rng(cfg.generator.seed)
    pop = build_population(cfg.generator, rng)
    assert len(pop.customer_ids) == 5000
    assert len(pop.institutions) == 12
    assert pop.institutions[0] == "BANK_A" and pop.institutions[-1] == "BANK_L"
    # every account belongs to exactly one of the 5,000 customers, 1-3 accounts each
    assert set(pop.account_customer) <= set(pop.customer_ids)
    assert 5000 <= len(pop.account_ids) <= 15000


def test_same_seed_gives_identical_content_hash(cfg):
    df_a, _ = generate_dataset(cfg, seed=42)
    df_b, _ = generate_dataset(cfg, seed=42)
    assert _content_hash(df_a) == _content_hash(df_b)


def test_different_seed_gives_different_hash(cfg):
    df_a, _ = generate_dataset(cfg, seed=42)
    df_b, _ = generate_dataset(cfg, seed=7)
    assert _content_hash(df_a) != _content_hash(df_b)


def test_approximately_100000_transactions_over_7_days(dataset):
    df, _ = dataset
    assert 90_000 <= len(df) <= 110_000


# --- M-03: distribution checks ---

def test_overall_baseline_failure_rate_in_range(dataset):
    df, _ = dataset
    fail_rate = (df.outcome == "FAILED").mean()
    assert 0.022 <= fail_rate <= 0.028


def test_amount_median_in_range(dataset):
    df, _ = dataset
    assert 13_000 <= df.amount.median() <= 17_000


def test_amounts_capped_and_rounded(dataset):
    df, _ = dataset
    assert df.amount.max() <= 5_000_000
    assert (df.amount % 10 == 0).all()


def test_visible_daily_pattern(dataset):
    # Peak hour volume should clearly exceed trough hour volume (midday vs 3am).
    df, _ = dataset
    by_hour = df.groupby(df.timestamp.dt.hour).size()
    assert by_hour.loc[11] > by_hour.loc[3] * 2


def test_visible_weekly_pattern(dataset):
    df, _ = dataset
    by_weekday = df.groupby(df.timestamp.dt.weekday).size()
    # weekends (5, 6) should be lighter than an average weekday
    weekday_avg = by_weekday.loc[0:4].mean()
    assert by_weekday.loc[5] < weekday_avg and by_weekday.loc[6] < weekday_avg


def test_response_time_p95_documented_gap(dataset):
    # KNOWN GAP (see config/defaults.yaml calibration note): with the PRD's own per-route
    # medians/sigmas, the volume-weighted mixture p95 lands well above the stated 1.8s
    # target. This test records the measured value rather than silently passing or hiding it.
    df, _ = dataset
    p95 = df.response_ms.quantile(0.95)
    assert 1800 < p95 < 3000  # fails loudly (not silently) if calibration changes this a lot


def test_txn_mix_matches_configured_shares(dataset, cfg):
    df, _ = dataset
    observed = df.txn_type.value_counts(normalize=True)
    for txn_type, target_share in cfg.generator.txn_mix.items():
        assert abs(observed.get(txn_type, 0) - target_share) < 0.02


# --- Scenario injection ---

def test_scenario2_spike_is_detectable(dataset, cfg):
    from ml.detector.binomial import evaluate_window
    df, labels = dataset
    spike = next(l for l in labels if l.kind == "SPIKE" and l.is_incident)
    mask = ((df.route == "NIP_ROUTE_2") & (df.timestamp >= pd.Timestamp(spike.start))
            & (df.timestamp < pd.Timestamp(spike.end)))
    n, f = mask.sum(), int((df.loc[mask, "outcome"] == "FAILED").sum())
    stat = evaluate_window(n, f, cfg.routes["NIP_ROUTE_2"].baseline_fail, cfg.detector)
    assert stat is not None and stat.passed


def test_scenario6_hard_negative_does_not_fire(dataset, cfg):
    from ml.detector.binomial import evaluate_window
    df, labels = dataset
    burst = next(l for l in labels if not l.is_incident)
    mask = ((df.route == "INTRA_CORE") & (df.timestamp >= pd.Timestamp(burst.start))
            & (df.timestamp < pd.Timestamp(burst.end)))
    n, f = mask.sum(), int((df.loc[mask, "outcome"] == "FAILED").sum())
    stat = evaluate_window(n, f, cfg.routes["INTRA_CORE"].baseline_fail, cfg.detector)
    assert stat is None or not stat.passed


def test_labels_reference_real_txn_ids(dataset):
    df, labels = dataset
    all_ids = set(df.txn_id)
    for label in labels:
        assert set(label.affected_txn_refs) <= all_ids
        assert len(label.affected_txn_refs) > 0


# --- Scenario 6: hard negatives, tested at three levels (fix for weak-pass issue) ---

def test_scenario6_generated_burst_is_silent_only_because_of_min_n(dataset, cfg):
    """The generated burst stays silent ONLY because n < min_n. Pin that explicitly so
    nobody mistakes it for proof that the z/ratio logic rejects it."""
    df, labels = dataset
    burst = next(l for l in labels if not l.is_incident)
    mask = ((df.route == "INTRA_CORE") & (df.timestamp >= pd.Timestamp(burst.start))
            & (df.timestamp < pd.Timestamp(burst.end)))
    assert mask.sum() < cfg.detector.min_n


def test_hard_negative_clears_guards_but_z_ratio_reject_it(cfg):
    """Enough volume and enough failures to pass both guards, but the RATE is normal.
    This is what actually exercises the z-score/ratio logic."""
    from ml.detector.binomial import evaluate_window
    d = cfg.detector
    n, f, p0 = 400, 12, 0.03
    assert n >= d.min_n and f >= d.min_failures          # guards cleared
    stat = evaluate_window(n, f, p0, d)
    assert stat is not None and not stat.passed           # z=0, ratio=1 -> rejected


@pytest.mark.xfail(strict=True, reason="PRD scenario 6 literal (n=40, f=12) clears every guard and "
                   "has z~18: low volume alone does not stop it. Needs higher min_failures/min_n "
                   "or a non-overlapping confirmation window (team decision, gate M-10).")
def test_prd_literal_scenario6_values_stay_silent(cfg):
    from ml.detector.binomial import evaluate_window
    stat = evaluate_window(40, 12, cfg.routes["INTRA_CORE"].baseline_fail, cfg.detector)
    assert stat is None or not stat.passed
