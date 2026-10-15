"""Scenario injection and ground-truth labels (ML PRD section 9 "Scenarios", Listing 4).

Each injected scenario mutates a copy of the baseline history inside a time window
and writes a label dict matching Listing 4's shape. Labels are never read by
production code -- they exist only for the evaluation notebooks (M-09, M-10).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Kind = Literal["SPIKE", "DELAY", "LATENCY", "OUTAGE", "NEGATIVE"]
Magnitude = Literal["WEAK", "MEDIUM", "STRONG"]


@dataclass(frozen=True)
class GroundTruthLabel:
    label_id: str
    kind: Kind
    start: pd.Timestamp
    end: pd.Timestamp
    target: dict            # {"route":..., "txn_type":..., "dest_institution":..., "channel":...}
    params: dict
    magnitude: Magnitude
    is_incident: bool
    affected_txn_refs: list  # txn_id values; written out as a separate refs file per Listing 4

    def to_dict(self, refs_file: str) -> dict:
        return {
            "label_id": self.label_id, "kind": self.kind,
            "start": self.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": self.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": self.target, "params": self.params, "magnitude": self.magnitude,
            "is_incident": self.is_incident, "affected_txn_refs_file": refs_file,
        }


def inject_spike(df: pd.DataFrame, *, route: str, start: pd.Timestamp, duration_minutes: int,
                 fail_probability: float, failure_code: str, magnitude: Magnitude,
                 rng: np.random.Generator, response_ms_multiplier: float = 3.0) -> tuple[pd.DataFrame, GroundTruthLabel]:
    """Raise the failure rate (and response time) on one route within [start, end).

    Re-rolls the outcome for every transaction on that route in the window, so the
    injected rate is exact regardless of what the baseline draw originally gave.
    """
    end = start + pd.Timedelta(minutes=duration_minutes)
    df = df.copy()
    mask = (df.route == route) & (df.timestamp >= start) & (df.timestamp < end)
    idx = df.index[mask]

    fail_draw = rng.random(len(idx)) < fail_probability
    df.loc[idx[fail_draw], "outcome"] = "FAILED"
    df.loc[idx[fail_draw], "failure_code"] = failure_code
    df.loc[idx[~fail_draw], "outcome"] = "SUCCESS"
    df.loc[idx[~fail_draw], "failure_code"] = None
    df.loc[idx, "response_ms"] = df.loc[idx, "response_ms"] * response_ms_multiplier

    label = GroundTruthLabel(
        label_id=f"inc-{uuid.uuid4().hex[:6]}", kind="SPIKE", start=start, end=end,
        target={"route": route, "txn_type": None, "dest_institution": None, "channel": None},
        params={"fail_probability": fail_probability, "failure_code": failure_code,
                "response_ms_multiplier": response_ms_multiplier},
        magnitude=magnitude, is_incident=True,
        affected_txn_refs=df.loc[idx, "txn_id"].tolist(),
    )
    return df, label


def inject_delayed_confirmation(df: pd.DataFrame, *, route: str, start: pd.Timestamp,
                                duration_minutes: int, delayed_share: float,
                                magnitude: Magnitude, rng: np.random.Generator) -> tuple[pd.DataFrame, GroundTruthLabel]:
    """Scenario 3: a share of route responses delayed beyond 90s, then succeed late.

    Modelled as PENDING at bucket close (M-11: must surface as timeout_rate or
    pending_rate, never fail_rate -- these transactions still succeed).
    """
    end = start + pd.Timedelta(minutes=duration_minutes)
    df = df.copy()
    mask = (df.route == route) & (df.timestamp >= start) & (df.timestamp < end)
    idx = df.index[mask]

    delay_draw = rng.random(len(idx)) < delayed_share
    delayed_idx = idx[delay_draw]
    df.loc[delayed_idx, "outcome"] = "PENDING"
    df.loc[delayed_idx, "response_ms"] = df.loc[delayed_idx, "response_ms"] + rng.uniform(90_000, 150_000, size=len(delayed_idx))

    label = GroundTruthLabel(
        label_id=f"inc-{uuid.uuid4().hex[:6]}", kind="DELAY", start=start, end=end,
        target={"route": route, "txn_type": None, "dest_institution": None, "channel": None},
        params={"delayed_share": delayed_share, "delay_threshold_s": 90},
        magnitude=magnitude, is_incident=True,
        affected_txn_refs=df.loc[delayed_idx, "txn_id"].tolist(),
    )
    return df, label


def inject_low_volume_burst(df: pd.DataFrame, *, route: str, start: pd.Timestamp,
                            duration_minutes: int, n_failures: int,
                            rng: np.random.Generator) -> tuple[pd.DataFrame, GroundTruthLabel]:
    """Scenario 6 (hard negative): a small burst of failures that must NOT raise a signal."""
    end = start + pd.Timedelta(minutes=duration_minutes)
    df = df.copy()
    mask = (df.route == route) & (df.timestamp >= start) & (df.timestamp < end)
    idx = df.index[mask]
    if len(idx) < n_failures:
        raise ValueError(f"only {len(idx)} candidate txns on {route} in window, need {n_failures}")

    chosen = rng.choice(idx, size=n_failures, replace=False)
    df.loc[chosen, "outcome"] = "FAILED"
    df.loc[chosen, "failure_code"] = "E_SYSTEM_ERROR"

    label = GroundTruthLabel(
        label_id=f"inc-{uuid.uuid4().hex[:6]}", kind="SPIKE", start=start, end=end,
        target={"route": route, "txn_type": None, "dest_institution": None, "channel": None},
        params={"n_failures": n_failures}, magnitude="WEAK", is_incident=False,
        affected_txn_refs=df.loc[chosen, "txn_id"].tolist(),
    )
    return df, label
