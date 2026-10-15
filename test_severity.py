import pytest
from ml.config import SeverityConfig, load_config
from packages.common.severity import band_for, severity

CFG = SeverityConfig()


def test_worked_vector():
    r = severity(427, 7.48, 31_245_000, 58, CFG)
    assert r.impact == pytest.approx(0.877, abs=1e-3)
    assert r.deviation == pytest.approx(0.748, abs=1e-3)
    assert r.value == pytest.approx(0.754, abs=1e-3)
    assert r.complaint_velocity == pytest.approx(0.580, abs=1e-3)
    assert r.score == pytest.approx(0.761, abs=1e-3)
    assert r.band == "HIGH"


def test_prd_literal_boundary_would_make_worked_vector_critical():
    # Documents why critical is 0.80 in config: with the PRD's stated 0.75 the vector is CRITICAL.
    literal = SeverityConfig(bands={"medium": 0.25, "high": 0.50, "critical": 0.75})
    assert severity(427, 7.48, 31_245_000, 58, literal).band == "CRITICAL"


@pytest.mark.parametrize("score,band", [
    (0.0, "LOW"), (0.2499, "LOW"), (0.25, "MEDIUM"), (0.4999, "MEDIUM"),
    (0.50, "HIGH"), (0.7999, "HIGH"), (0.80, "CRITICAL"), (1.0, "CRITICAL")])
def test_band_boundaries_both_sides(score, band):
    assert band_for(score, CFG) == band


def test_zero_everything():
    r = severity(0, 0.0, 0.0, 0, CFG)
    assert r.score == 0.0 and r.band == "LOW"


def test_zero_complaints_still_scores_from_other_components():
    r = severity(427, 7.48, 31_245_000, 0, CFG)
    assert r.complaint_velocity == 0.0
    assert 0 < r.score < 0.761


def test_very_large_values_clamp_to_one():
    r = severity(10**9, 1000.0, 10**15, 10**6, CFG)
    assert (r.impact, r.deviation, r.value, r.complaint_velocity) == (1.0, 1.0, 1.0, 1.0)
    assert r.score == pytest.approx(1.0)


def test_negative_inputs_do_not_go_below_zero():
    assert severity(-5, -1.0, -100.0, -3, CFG).score == 0.0


def test_yaml_config_matches_dataclass_defaults():
    assert load_config().severity == SeverityConfig()
