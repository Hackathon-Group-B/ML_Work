import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ml.config import Config, DetectorConfig
from ml.detector.binomial import (InMemoryStreakStore, build_signal, clamp_baseline,
                                  evaluate_window, update_streak)

CFG = DetectorConfig()
SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "ml/schemas/anomaly_signal.schema.json").read_text())


def test_worked_vector():
    s = evaluate_window(2284, 427, 0.025, CFG)
    assert s.p_hat == pytest.approx(0.187, abs=1e-3)
    assert round(s.z, 1) == 49.6
    assert round(s.ratio, 2) == 7.48
    assert s.passed


@pytest.mark.parametrize("n,f", [(29, 20), (100, 9), (0, 0)])
def test_too_small_returns_none(n, f):
    assert evaluate_window(n, f, 0.025, CFG) is None


def test_normal_traffic_does_not_pass():
    s = evaluate_window(400, 10, 0.025, CFG)   # exactly baseline rate
    assert s is not None and not s.passed


def test_requires_both_z_and_ratio():
    # High z but ratio < 3: large n, modest elevation 2.5% -> 6.5% (ratio 2.6, z ~ 20)
    s = evaluate_window(20000, 1300, 0.025, CFG)
    assert s.z >= CFG.z_min and s.ratio < CFG.ratio_min and not s.passed


def test_single_window_never_fires_alone():
    s = evaluate_window(40, 12, 0.025, CFG)
    assert not update_streak("k", s, InMemoryStreakStore(), CFG)


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN SPEC GAP (gate M-10): 5-min windows are evaluated every 30 s, so the same low-volume "
    "burst sits in ~10 consecutive windows and passes the 2-consecutive rule. Needs a guard "
    "(higher min_n/min_failures, or non-overlapping confirmation window). Decide via sensitivity grid."))
def test_scenario6_low_volume_burst_stays_silent_over_overlapping_ticks():
    store = InMemoryStreakStore()
    fired = [update_streak("seg", evaluate_window(40, 12, 0.025, CFG), store, CFG) for _ in range(10)]
    assert not any(fired)


def test_baseline_floor_and_cap():
    assert clamp_baseline(0.0, CFG) == 0.005
    assert clamp_baseline(0.9, CFG) == 0.5
    s = evaluate_window(100, 10, 0.0, CFG)       # zero baseline must not give infinite z
    assert s is not None and s.z < float("inf")


def test_below_baseline_never_passes():
    s = evaluate_window(1000, 10, 0.05, CFG)
    assert s.z < 0 and not s.passed


def test_streak_needs_two_consecutive_and_resets():
    good = evaluate_window(2284, 427, 0.025, CFG)
    store = InMemoryStreakStore()
    assert not update_streak("seg", good, store, CFG)      # 1st
    assert update_streak("seg", good, store, CFG)          # 2nd fires
    assert update_streak("seg", good, store, CFG)          # keeps firing while anomalous
    assert not update_streak("seg", None, store, CFG)      # too-small window resets
    assert not update_streak("seg", good, store, CFG)      # back to 1
    assert store.get("other") == 0                         # segments are independent


def test_streak_state_survives_store_swap():
    # Restart safety: a new worker with the same persisted store continues the streak.
    good = evaluate_window(2284, 427, 0.025, CFG)
    store = InMemoryStreakStore()
    update_streak("seg", good, store, CFG)
    assert update_streak("seg", good, store, CFG)


def test_signal_validates_against_schema_and_matches_listing_2():
    stat = evaluate_window(2284, 427, 0.025, CFG)
    t = datetime(2026, 10, 20, 14, 32, 5, tzinfo=timezone.utc)
    sig = build_signal(
        stat=stat, metric="fail_rate",
        segment={"channel": None, "txn_type": "INTERBANK_TRANSFER",
                 "dest_institution": None, "route": "NIP_ROUTE_2"},
        window_start=datetime(2026, 10, 20, 14, 18, tzinfo=timezone.utc),
        window_end=datetime(2026, 10, 20, 14, 32, tzinfo=timezone.utc),
        detected_at=t, amount_affected_ngn=31_245_000, complaints_last_15min=58,
        top_failure_code="E_BENEFICIARY_TIMEOUT", cfg=Config())
    Draft202012Validator(SCHEMA).validate(sig)
    assert sig["z_score"] == 49.6 and sig["ratio"] == 7.48
    assert sig["n_txn"] == 2284 and sig["n_failed"] == 427
    assert sig["severity_score"] == 0.76 and sig["severity"] == "HIGH"


# --- Sustained-evidence rule (evaluate_sustained) ---

from dataclasses import replace as _replace

from ml.config import load_config
from ml.detector.binomial import evaluate_sustained, evaluate_window


def _sustained_cfg(**kw):
    return _replace(load_config().detector, **kw)


def test_sustained_rejects_cluster_crammed_into_one_slice():
    """12 failures all in one slice: the pooled window looks alarming, the other
    slices are at baseline, so it must NOT fire."""
    cfg = _sustained_cfg(z_min=4.0, confirm_parts=3, confirm_min_parts=3)
    parts = [(83, 12), (83, 0), (84, 1)]               # pooled n=250, f=13, p0=1%
    pooled = evaluate_window(250, 13, 0.01, cfg)
    assert pooled.passed                                # the pooled test alone WOULD fire
    assert not evaluate_sustained(parts, 0.01, cfg).passed


def test_sustained_accepts_incident_spread_across_all_slices():
    cfg = _sustained_cfg(confirm_parts=3, confirm_min_parts=3)
    parts = [(60, 11), (60, 12), (60, 11)]              # ~19% failures throughout, p0=2.5%
    assert evaluate_sustained(parts, 0.025, cfg).passed


def test_sustained_disabled_matches_plain_window():
    cfg = _sustained_cfg(confirm_parts=1, confirm_min_parts=1)
    parts = [(83, 12), (83, 0), (84, 1)]
    assert evaluate_sustained(parts, 0.01, cfg) == evaluate_window(250, 13, 0.01, cfg)


def test_sustained_slice_needs_minimum_failures():
    """A slice with a high rate but only 2 failures is not counted as elevated."""
    cfg = _sustained_cfg(z_min=4.0, confirm_parts=3, confirm_min_parts=3, part_min_failures=3)
    parts = [(60, 12), (60, 12), (20, 2)]              # third slice: 10% rate but just 2 failures
    assert not evaluate_sustained(parts, 0.025, cfg).passed


def test_sustained_returns_none_when_window_too_small():
    cfg = _sustained_cfg(confirm_parts=3, confirm_min_parts=3)
    assert evaluate_sustained([(5, 1), (5, 1), (5, 1)], 0.025, cfg) is None


def test_shipped_defaults_use_the_sweep_chosen_rule():
    d = load_config().detector
    assert d.z_min == 5.0 and d.confirm_parts == 3 and d.confirm_min_parts == 3
