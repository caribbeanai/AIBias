"""60-second quickstart."""
from aibias import TextBiasDetector, BiasMitigator

text = (
    "The chairman, an old white man, dismissed the female engineer's "
    "concerns. The young Asian intern stayed silent."
)

# Detect
report = TextBiasDetector().analyze(text)
print(report.summary())

# Mitigate
clean, mreport = BiasMitigator().neutralize(text)
print("\n--- Neutralized ---\n", clean)
print(mreport.summary())
