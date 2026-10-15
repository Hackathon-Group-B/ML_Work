"""Per-transaction simulation: type, route, amount, response time, outcome (ML PRD section 9).

generate_history() is the main entry point: given a seed and config, it returns a
fully reproducible transactions DataFrame for the whole history window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import Config, GeneratorConfig

TXN_TYPES = ["INTERBANK_TRANSFER", "INTRABANK_TRANSFER", "AIRTIME", "BILL_PAYMENT",
            "CARD", "DATA", "QR", "CASH_WITHDRAWAL"]


def _route_for_txn_type(cfg: Config, txn_type: str) -> str:
    """The route(s) available for a given txn_type (ML PRD Table 4)."""
    candidates = [name for name, spec in cfg.routes.items() if txn_type in spec.used_by]
    if not candidates:
        raise ValueError(f"no route configured for txn_type {txn_type}")
    return candidates


def _sample_amounts(n: int, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Log-normal amounts, median cfg.amount_median_ngn, capped and rounded (ML PRD Table 3)."""
    mu = np.log(cfg.amount_median_ngn)
    amounts = rng.lognormal(mean=mu, sigma=cfg.amount_sigma, size=n)
    amounts = np.minimum(amounts, cfg.amount_cap_ngn)
    amounts = np.round(amounts / cfg.amount_round_to_ngn) * cfg.amount_round_to_ngn
    return amounts


def _sample_response_ms(n: int, median_ms: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Log-normal response time per route: median_ms * exp(sigma * N(0,1)) (ML PRD section 9)."""
    z = rng.normal(0, 1, size=n)
    return median_ms * np.exp(sigma * z)


def _pick_failure_code(cfg: Config, rng: np.random.Generator, n: int) -> np.ndarray:
    codes = list(cfg.failure_codes.keys())
    shares = np.array([cfg.failure_codes[c].share for c in codes])
    shares = shares / shares.sum()
    return rng.choice(codes, size=n, p=shares)


def _debited_for_code(code: str, cfg: Config, rng: np.random.Generator) -> bool:
    spec = cfg.failure_codes[code].debited
    if spec == "yes":
        return True
    if spec == "never":
        return False
    # "sometimes:<p>"
    p = float(spec.split(":")[1])
    return bool(rng.random() < p)


def generate_history(cfg: Config, population, hours: pd.DatetimeIndex,
                     counts_per_hour: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the full transaction history. Deterministic for a fixed seed and config.

    One rng is threaded through in a fixed order (timing -> type -> route -> amount ->
    response time -> outcome) so the same seed always reproduces the same table.
    """
    gcfg = cfg.generator
    n_accounts = len(population.account_ids)
    txn_types = list(gcfg.txn_mix.keys())
    txn_type_p = np.array(list(gcfg.txn_mix.values()))
    txn_type_p = txn_type_p / txn_type_p.sum()

    rows = []
    txn_seq = 1
    for hour_start, n_this_hour in zip(hours, counts_per_hour):
        if n_this_hour == 0:
            continue
        # Spread arrivals uniformly within the hour (Poisson process approximation).
        offsets = np.sort(rng.integers(0, 3600, size=n_this_hour))
        timestamps = hour_start + pd.to_timedelta(offsets, unit="s")

        acct_idx = rng.integers(0, n_accounts, size=n_this_hour)
        source_accounts = population.account_ids[acct_idx]
        customers = population.account_customer[acct_idx]

        types = rng.choice(txn_types, size=n_this_hour, p=txn_type_p)
        routes = np.array([rng.choice(_route_for_txn_type(cfg, t)) for t in types])
        dest_institutions = rng.choice(population.institutions, size=n_this_hour)

        amounts = _sample_amounts(n_this_hour, gcfg, rng)

        response_ms = np.empty(n_this_hour)
        outcomes = np.empty(n_this_hour, dtype=object)
        failure_codes = np.array([None] * n_this_hour, dtype=object)
        debited_flags = np.zeros(n_this_hour, dtype=bool)

        for route_name in np.unique(routes):
            mask = routes == route_name
            spec = cfg.routes[route_name]
            response_ms[mask] = _sample_response_ms(mask.sum(), spec.median_ms, spec.sigma, rng)
            fail_draw = rng.random(mask.sum()) < spec.baseline_fail
            idx = np.where(mask)[0]
            for i, failed in zip(idx, fail_draw):
                if failed:
                    code = _pick_failure_code(cfg, rng, 1)[0]
                    failure_codes[i] = code
                    debited_flags[i] = _debited_for_code(code, cfg, rng)
                    outcomes[i] = "FAILED"
                else:
                    outcomes[i] = "SUCCESS"
                    debited_flags[i] = True

        rows.append(pd.DataFrame({
            "txn_id": [f"ZT{str(n).zfill(6)}" for n in range(txn_seq, txn_seq + n_this_hour)],
            "customer_id": customers, "source_account": source_accounts,
            "timestamp": timestamps, "txn_type": types, "route": routes,
            "dest_institution": dest_institutions, "amount": amounts,
            "response_ms": response_ms, "outcome": outcomes,
            "failure_code": failure_codes, "debited": debited_flags,
        }))
        txn_seq += n_this_hour

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
