import json
from pathlib import Path

from aibias import BiasAuditor


def test_audit_text_only(tmp_path):
    a = BiasAuditor()
    report = a.audit(texts=[
        "The engineer is a man named Carlos. The nurse is a woman.",
        "She was emotional, he was rational.",
    ])
    d = report.to_dict()
    assert "overall_bias_index" in d
    assert d["text_report"] is not None


def test_audit_classifier_metrics_flag_di():
    a = BiasAuditor()
    report = a.audit(
        classifier_outputs={
            "positive_rates": {"m": 0.7, "f": 0.3},
            "privileged_group": "m",
        }
    )
    severities = [f.severity for f in report.findings]
    assert "high" in severities  # DI = 0.43 violates 80% rule


def test_audit_save_artifacts(tmp_path):
    a = BiasAuditor()
    report = a.audit(texts=["Hello world."])
    paths = a.save(report, tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    json.loads(paths["json"].read_text())
