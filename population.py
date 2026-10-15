"""Customer, account and institution population for the synthetic generator.

Pure, seeded, reproducible: build_population(seed) always returns identical data
for the same seed and config (ML PRD section 9, M-02).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.config import GeneratorConfig


@dataclass(frozen=True)
class Population:
    customer_ids: np.ndarray          # shape (n_customers,) e.g. "CUST00001"
    account_customer: np.ndarray      # shape (n_accounts,) -> customer_id for each account
    account_ids: np.ndarray           # shape (n_accounts,) e.g. "ACC000001"
    institutions: tuple               # ("BANK_A", ..., "BANK_L")


def build_population(cfg: GeneratorConfig, rng: np.random.Generator) -> Population:
    customer_ids = np.array([f"CUST{str(i).zfill(6)}" for i in range(1, cfg.n_customers + 1)])

    # Each customer has 1..3 accounts (inclusive), drawn with the seeded rng.
    n_accounts_per_customer = rng.integers(
        cfg.accounts_per_customer_min, cfg.accounts_per_customer_max + 1, size=cfg.n_customers)

    account_customer = np.repeat(customer_ids, n_accounts_per_customer)
    account_ids = np.array([f"ACC{str(i).zfill(7)}" for i in range(1, len(account_customer) + 1)])

    institutions = tuple(chr(ord("A") + i) for i in range(cfg.n_institutions))
    institutions = tuple(f"BANK_{letter}" for letter in institutions)

    return Population(customer_ids, account_customer, account_ids, institutions)
