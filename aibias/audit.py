"""End-to-end bias audit orchestrator.

Combines text + image + dataset-classification metrics into a single report
suitable for compliance documentation (EU AI Act risk assessments, NIST AI RMF,
ISO/IEC 42001 fairness audits).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .text import TextBiasDetector
from .image import ImageBiasDetector, AttributeClassifier
from .mitigate import BiasMitigator
from .metrics import (
    statistical_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
    equalized_odds_difference,
    bias_index,
)


@dataclass
class AuditFinding:
    severity: str
    axis: str
    description: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


@dataclass
class AuditReport:
    timestamp: str
    inputs_summary: dict[str, Any]
    text_report: dict | None = None
    image_report: dict | None = None
    classifier_metrics: dict | None = None
    findings: list[AuditFinding] = field(default_factory=list)
    overall_bias_index: dict | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# AI Bias Audit Report",
            f"_Generated: {self.timestamp}_",
            "",
            "## Inputs",
        ]
        for k, v in self.inputs_summary.items():
            lines.append(f"- **{k}**: {v}")
        if self.overall_bias_index:
            lines += [
                "",
                "## Overall Bias Index",
                f"**{self.overall_bias_index['overall']}/100** "
                f"(confidence {self.overall_bias_index['confidence']})",
                "",
                "| Axis | Score |",
                "|---|---|",
            ]
            for axis, score in self.overall_bias_index["by_axis"].items():
                lines.append(f"| {axis} | {score} |")
        if self.findings:
            lines += ["", "## Findings"]
            for f in self.findings:
                lines.append(
                    f"- **[{f.severity}]** ({f.axis}) {f.description}"
                    + (f" — `{f.metric}={f.value}`" if f.metric else "")
                )
        if self.text_report:
            lines += ["", "## Text Analysis", "```json",
                      json.dumps(self.text_report.get("bias_index"), indent=2), "```"]
        if self.image_report:
            lines += ["", "## Image Analysis", "```json",
                      json.dumps(self.image_report.get("bias_index"), indent=2), "```"]
        if self.classifier_metrics:
            lines += ["", "## Classifier Fairness", "```json",
                      json.dumps(self.classifier_metrics, indent=2), "```"]
        if self.recommendations:
            lines += ["", "## Recommendations"]
            lines += [f"1. {r}" for r in self.recommendations]
        return "\n".join(lines)


class BiasAuditor:
    """Aggregate detector that builds a unified audit report."""

    def __init__(
        self,
        text_detector: TextBiasDetector | None = None,
        image_detector: ImageBiasDetector | None = None,
        mitigator: BiasMitigator | None = None,
    ):
        self.text = text_detector or TextBiasDetector()
        self.image = image_detector or ImageBiasDetector()
        self.mitigator = mitigator or BiasMitigator()

    def audit(
        self,
        texts: Sequence[str] | None = None,
        image_paths: Sequence[str] | None = None,
        classifier_outputs: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Run end-to-end audit.

        classifier_outputs: optional dict with keys:
            positive_rates: {group: rate}
            tpr_by_group: {group: rate}
            fpr_by_group: {group: rate}
            privileged_group: str
        """
        ts = datetime.now(timezone.utc).isoformat()
        findings: list[AuditFinding] = []
        axis_scores: dict[str, float] = {}
        sample_n = 0

        text_report = None
        if texts:
            combined = "\n\n".join(texts)
            tr = self.text.analyze(combined)
            text_report = tr.to_dict()
            for ax, sc in tr.axis_scores.items():
                axis_scores[f"text:{ax}"] = max(axis_scores.get(f"text:{ax}", 0), sc)
            for f in tr.findings:
                findings.append(AuditFinding(
                    severity=f.severity, axis=f.axis, description=f.message,
                    metric=f.kind, value=round(f.score, 3),
                ))
            sample_n += tr.token_count

        image_report = None
        if image_paths:
            ir = self.image.analyze(image_paths)
            image_report = ir.to_dict()
            for ax, sc in ir.axis_scores.items():
                axis_scores[f"image:{ax}"] = sc
            for f in ir.findings:
                findings.append(AuditFinding(
                    severity="medium", axis="image", description=f,
                ))
            sample_n += ir.n_images * 50

        classifier_metrics = None
        if classifier_outputs:
            classifier_metrics = self._classifier_metrics(classifier_outputs, findings, axis_scores)

        overall = bias_index(axis_scores, sample_size=sample_n).to_dict()

        recs = self._recommendations(findings, axis_scores)

        return AuditReport(
            timestamp=ts,
            inputs_summary={
                "n_texts": len(texts) if texts else 0,
                "n_images": len(image_paths) if image_paths else 0,
                "classifier_provided": bool(classifier_outputs),
            },
            text_report=text_report,
            image_report=image_report,
            classifier_metrics=classifier_metrics,
            findings=findings,
            overall_bias_index=overall,
            recommendations=recs,
        )

    # ----------------------------------------------------------------- helpers
    def _classifier_metrics(self, co, findings, axis_scores) -> dict:
        priv = co["privileged_group"]
        out: dict[str, Any] = {}
        if "positive_rates" in co:
            spd = statistical_parity_difference(co["positive_rates"], priv)
            di = disparate_impact(co["positive_rates"], priv)
            out["statistical_parity_difference"] = spd
            out["disparate_impact"] = di
            for g, ratio in di.items():
                if ratio < 0.8 or ratio > 1.25:
                    findings.append(AuditFinding(
                        severity="high", axis="classifier",
                        description=f"Disparate impact for '{g}' violates 80% rule.",
                        metric="DI", value=round(ratio, 3), threshold=0.8,
                    ))
                    axis_scores["classifier:disparate_impact"] = max(
                        axis_scores.get("classifier:disparate_impact", 0),
                        min(1.0, abs(1 - ratio)),
                    )
        if "tpr_by_group" in co and "fpr_by_group" in co:
            out["equalized_odds"] = equalized_odds_difference(
                co["tpr_by_group"], co["fpr_by_group"], priv
            )
            for g, gap in out["equalized_odds"]["tpr_gap"].items():
                if abs(gap) > 0.1:
                    findings.append(AuditFinding(
                        severity="high" if abs(gap) > 0.2 else "medium",
                        axis="classifier",
                        description=f"Equal-opportunity gap for '{g}': TPR diff {gap:.3f}",
                        metric="EOD", value=round(gap, 3), threshold=0.1,
                    ))
                    axis_scores["classifier:eo"] = max(
                        axis_scores.get("classifier:eo", 0), min(1.0, abs(gap) * 2)
                    )
        elif "tpr_by_group" in co:
            out["equal_opportunity_difference"] = equal_opportunity_difference(
                co["tpr_by_group"], priv
            )
        return out

    def _recommendations(self, findings, axis_scores) -> list[str]:
        recs = []
        sev = Counter_severity(findings)
        if sev["high"]:
            recs.append("Block-list slurs/pejoratives detected — remove before deployment.")
            recs.append("Run counterfactual data augmentation (BiasMitigator.cda_augment) on training data.")
        if any(a.startswith("text:gender") for a in axis_scores):
            recs.append("Use neutral occupational language; rewrite gendered job titles.")
        if any(a.startswith("image:") for a in axis_scores):
            recs.append("Diversify image-generation prompts; over-sample under-represented groups.")
        if any(a.startswith("classifier:disparate_impact") for a in axis_scores):
            recs.append("Apply post-processing fairness correction (reweighing, calibrated equalized odds).")
        if not recs:
            recs.append("No critical bias signals found; continue periodic monitoring.")
        recs.append("Document this audit per NIST AI RMF GOVERN-1.1 / EU AI Act Art. 10(2)(f).")
        return recs

    # ------------------------------------------------------------------- I/O
    def save(self, report: AuditReport, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "bias_audit.json"
        md_path = out / "bias_audit.md"
        json_path.write_text(report.to_json())
        md_path.write_text(report.to_markdown())
        return {"json": json_path, "markdown": md_path}


def Counter_severity(findings):
    from collections import Counter
    return Counter(f.severity for f in findings)
