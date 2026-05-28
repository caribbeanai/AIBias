import math
import pytest

from aibias.metrics import (
    statistical_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
    equalized_odds_difference,
    representation_ratio,
    normalized_pointwise_mutual_information,
    log_probability_bias_score,
    bias_index,
    theil_index,
    weat_effect_size,
)


def test_statistical_parity_difference():
    rates = {"m": 0.6, "f": 0.4}
    out = statistical_parity_difference(rates, "m")
    assert out == {"f": pytest.approx(-0.2)}


def test_disparate_impact_eighty_rule():
    di = disparate_impact({"m": 0.5, "f": 0.4}, "m")
    assert di["f"] == pytest.approx(0.8)


def test_equal_opportunity_difference():
    tpr = {"m": 0.9, "f": 0.7}
    out = equal_opportunity_difference(tpr, "m")
    assert out["f"] == pytest.approx(-0.2)


def test_equalized_odds_bundle():
    out = equalized_odds_difference({"m": 0.9, "f": 0.7}, {"m": 0.1, "f": 0.2}, "m")
    assert set(out) == {"tpr_gap", "fpr_gap"}


def test_representation_with_baseline():
    obs = representation_ratio({"a": 60, "b": 40}, baseline={"a": 0.5, "b": 0.5})
    assert obs["a"] == pytest.approx(1.2)
    assert obs["b"] == pytest.approx(0.8)


def test_npmi_perfect_association():
    # joint == marg_x == marg_y == total -> NPMI = 1
    assert normalized_pointwise_mutual_information(10, 10, 10, 10) == pytest.approx(1.0, abs=1e-9) or \
           math.isnan(normalized_pointwise_mutual_information(10, 10, 10, 10))


def test_npmi_independence_zero():
    # When joint = marg_x*marg_y/total, PMI = 0 → NPMI = 0
    v = normalized_pointwise_mutual_information(5, 10, 10, 20)
    assert abs(v) < 1e-9


def test_log_probability_bias_score_positive():
    s = log_probability_bias_score(0.8, 0.2)
    assert s > 0


def test_bias_index_zero():
    bi = bias_index({})
    assert bi.overall == 0.0


def test_bias_index_caps_at_100():
    bi = bias_index({"a": 1.0, "b": 1.0, "c": 1.0})
    assert bi.overall == pytest.approx(100.0)


def test_theil_index_equal_values():
    assert theil_index([1, 1, 1, 1]) == pytest.approx(0.0)


def test_weat_runs():
    # simple 2D vectors
    X = [[1.0, 0.0], [0.9, 0.1]]
    Y = [[-1.0, 0.0], [-0.9, -0.1]]
    A = [[1.0, 0.0]]
    B = [[-1.0, 0.0]]
    d = weat_effect_size(X, Y, A, B)
    assert d > 0
