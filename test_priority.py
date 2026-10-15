"""A-07: priority rules are deterministic and raise-only (AI PRD 6.2)."""
import itertools

import pytest

from ai.complaints.priority import ORDER, LinkedTxn, apply_priority_rules
from ml.config import load_config

CFG = load_config().complaints
CATS = ["FAILED_TRANSACTION", "PENDING_TRANSACTION", "DEBITED_NOT_CREDITED", "REVERSAL",
        "UNAUTHORIZED_TRANSACTION", "TRANSFER_DELAY", "REFUND_ISSUE", "GENERAL_ENQUIRY"]


def test_unauthorized_is_critical_and_routed_to_risk():
    r = apply_priority_rules("LOW", "UNAUTHORIZED_TRANSACTION", CFG)
    assert r.priority == "CRITICAL" and r.route_to_risk and "unauthorized" in r.fired


def test_other_categories_are_not_routed_to_risk():
    assert not apply_priority_rules("LOW", "FAILED_TRANSACTION", CFG).route_to_risk


def test_debited_unreversed_over_30_minutes_is_high():
    t = LinkedTxn(debited=True, reversed=False, age_minutes=45)
    assert apply_priority_rules("LOW", "DEBITED_NOT_CREDITED", CFG, linked=t).priority == "HIGH"


@pytest.mark.parametrize("txn", [
    LinkedTxn(debited=True, reversed=False, age_minutes=30),    # exactly 30: not "over"
    LinkedTxn(debited=True, reversed=True, age_minutes=120),    # already reversed
    LinkedTxn(debited=False, reversed=False, age_minutes=120),  # never debited
])
def test_debited_rule_does_not_fire_outside_its_condition(txn):
    assert apply_priority_rules("LOW", "DEBITED_NOT_CREDITED", CFG, linked=txn).priority == "LOW"


def test_high_value_threshold_is_inclusive():
    assert apply_priority_rules("LOW", "FAILED_TRANSACTION", CFG, extracted_amount_ngn=500_000).priority == "HIGH"
    assert apply_priority_rules("LOW", "FAILED_TRANSACTION", CFG, extracted_amount_ngn=499_999).priority == "LOW"


def test_linked_amount_beats_extracted_amount():
    t = LinkedTxn(debited=False, reversed=False, age_minutes=1, amount_ngn=1_000_000)
    r = apply_priority_rules("LOW", "FAILED_TRANSACTION", CFG, linked=t, extracted_amount_ngn=1_000)
    assert r.priority == "HIGH"


@pytest.mark.parametrize("incident,expected", [("LOW", "MEDIUM"), ("MEDIUM", "MEDIUM"),
                                               ("HIGH", "HIGH"), ("CRITICAL", "HIGH")])
def test_open_incident_floor(incident, expected):
    r = apply_priority_rules("LOW", "FAILED_TRANSACTION", CFG, open_incident_priority=incident)
    assert r.priority == expected


def test_general_enquiry_without_link_stays_low_but_never_lowers():
    assert apply_priority_rules("LOW", "GENERAL_ENQUIRY", CFG).priority == "LOW"
    assert apply_priority_rules("HIGH", "GENERAL_ENQUIRY", CFG).priority == "HIGH"


def test_fired_lists_only_rules_that_actually_raised_priority():
    r = apply_priority_rules("CRITICAL", "UNAUTHORIZED_TRANSACTION", CFG)
    assert r.priority == "CRITICAL" and r.fired == ()          # nothing needed raising


def test_raise_only_over_a_grid_of_inputs():
    """No combination of inputs may ever produce a priority below the model's."""
    txns = [None, LinkedTxn(True, False, 90, 600_000), LinkedTxn(False, True, 5, 100)]
    for model_p, cat, txn, amt, inc in itertools.product(
            ORDER, CATS, txns, [None, 10.0, 900_000.0], [None, "LOW", "MEDIUM", "HIGH", "CRITICAL"]):
        r = apply_priority_rules(model_p, cat, CFG, linked=txn, extracted_amount_ngn=amt,
                                 open_incident_priority=inc)
        assert ORDER.index(r.priority) >= ORDER.index(model_p), (model_p, cat, txn, amt, inc)
