from aibias import BiasMitigator


def test_neutralize_gendered_titles():
    m = BiasMitigator()
    clean, report = m.neutralize("The chairman thanked the policemen and the stewardess.")
    assert "chair" in clean.lower()
    assert "police officers" in clean.lower() or "police officer" in clean.lower()
    assert "flight attendant" in clean.lower()
    assert sum(report.replacements.values()) >= 3


def test_redacts_slurs():
    m = BiasMitigator()
    clean, report = m.neutralize("That movie was retarded.")
    assert "[REDACTED]" in clean
    assert "retarded" in report.removed_terms


def test_counterfactual_swap_reciprocal():
    m = BiasMitigator()
    out = m.counterfactual_swap("He kissed his wife.")
    assert "she" in out.lower() and "husband" in out.lower()


def test_balanced_resample():
    items = [("m", 1)] * 10 + [("f", 1)] * 4
    out = BiasMitigator.balanced_resample(items, group_key=lambda x: x[0], seed=1)
    males = sum(1 for x in out if x[0] == "m")
    females = sum(1 for x in out if x[0] == "f")
    assert males == females
