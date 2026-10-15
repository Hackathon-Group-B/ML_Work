"""A-04 / A-06 plumbing and regression checks for the fallback classifier and its data."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from ai.complaints.classifier import FallbackClassifier, clean_for_model, fallback_summary
from ai.complaints.dataset import CATEGORIES, _norm_key, add_typos, build_datasets


@pytest.fixture(scope="module")
def data():
    return build_datasets(seed=7)


@pytest.fixture(scope="module")
def trained(data):
    train, _ = data
    return FallbackClassifier().fit([c.text for c in train], [c.category for c in train])


# --- the data meets the PRD's size and separation requirements ----------------------
def test_dataset_sizes_meet_the_prd(data):
    train, heldout = data
    assert len(train) >= 800 and len(heldout) >= 240
    for cat in CATEGORIES:
        assert sum(c.category == cat for c in train) >= 100
        assert sum(c.category == cat for c in heldout) >= 30


def test_train_and_heldout_share_no_template_and_no_text(data):
    train, heldout = data
    assert not {c.template_id for c in train} & {c.template_id for c in heldout}
    assert not {_norm_key(c.text) for c in train} & {_norm_key(c.text) for c in heldout}


def test_heldout_covers_english_pidgin_and_typos(data):
    _, heldout = data
    assert {c.lang for c in heldout} == {"en", "pcm"}
    assert {c.variant for c in heldout} == {"clean", "typo"}


def test_add_typos_changes_the_text():
    rng = np.random.default_rng(1)
    assert add_typos("transfer failed again today please", rng, rate=1.0) != "transfer failed again today please"


def test_dataset_is_reproducible():
    a, _ = build_datasets(seed=7)
    b, _ = build_datasets(seed=7)
    assert [c.text for c in a] == [c.text for c in b]


# --- the classifier --------------------------------------------------------------
def test_predictions_are_valid_categories_with_probabilities(trained, data):
    _, heldout = data
    for cat, conf in trained.predict_many([c.text for c in heldout[:40]]):
        assert cat in CATEGORIES and 0.0 <= conf <= 1.0


def test_obvious_messages_are_classified_correctly(trained):
    assert trained.predict("I want a refund for this payment")[0] == "REFUND_ISSUE"
    assert trained.predict("I did not make this transfer")[0] == "UNAUTHORIZED_TRANSACTION"
    assert trained.predict("How do I check my transfer limit?")[0] == "GENERAL_ENQUIRY"


def test_pii_never_reaches_the_model_input():
    cleaned = clean_for_model("call 08031234567 acct 0123456789 a@b.com ref ZT839102 sent N85,000")
    for leaked in ("0803", "0123456789", "a@b.com", "839102", "85,000"):
        assert leaked not in cleaned


def test_latency_p95_is_under_one_second(trained, data):
    """A-06: fallback p95 under 1 s."""
    _, heldout = data
    lat = trained.latencies_ms([c.text for c in heldout[:60]])
    assert np.percentile(lat, 95) < 1000


def test_fallback_summary_format():
    s = fallback_summary("REVERSAL", "ZT839102", "x" * 300)
    assert s == "REVERSAL for ZT839102: " + "x" * 140
    assert fallback_summary("REVERSAL", None, "hello").startswith("REVERSAL for unlinked: ")


# --- accuracy gates ----------------------------------------------------------------
def _heldout_accuracy(trained, data):
    _, heldout = data
    preds = [p[0] for p in trained.predict_many([c.text for c in heldout])]
    return accuracy_score([c.category for c in heldout], preds)


def test_accuracy_regression_floor(trained, data):
    """Not a quality claim: stops the classifier silently getting worse than today's 0.74."""
    assert _heldout_accuracy(trained, data) >= 0.70


@pytest.mark.xfail(strict=True, reason="A-06 proposes fallback accuracy >= 0.80. Measured about 0.74 on "
                   "template-disjoint held-out data: the 10 training templates per category are too "
                   "few. Needs more varied training text (LLM paraphrase or real messages).")
def test_fallback_accuracy_target_a06(trained, data):
    assert _heldout_accuracy(trained, data) >= 0.80
