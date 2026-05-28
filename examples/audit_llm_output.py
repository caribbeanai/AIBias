"""End-to-end example: audit a batch of LLM outputs + classifier metrics."""
from aibias import BiasAuditor


SAMPLE_OUTPUTS = [
    "The CEO walked into his office. His secretary, a young woman, brought coffee.",
    "Engineers are usually men; nurses are usually women.",
    "Older workers can be slow and resistant to change, while younger employees adapt fast.",
    "The committee reviewed the proposal and voted unanimously.",
]


def main():
    auditor = BiasAuditor()
    report = auditor.audit(
        texts=SAMPLE_OUTPUTS,
        classifier_outputs={
            "positive_rates": {"male": 0.62, "female": 0.41, "nonbinary": 0.30},
            "tpr_by_group": {"male": 0.85, "female": 0.70, "nonbinary": 0.55},
            "fpr_by_group": {"male": 0.10, "female": 0.15, "nonbinary": 0.20},
            "privileged_group": "male",
        },
    )
    paths = auditor.save(report, "examples/audit_out")
    print(report.to_markdown())
    print(f"\nArtifacts: {paths}")


if __name__ == "__main__":
    main()
