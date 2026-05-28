"""AIBias: detection, quantification, and mitigation of bias in AI-generated content."""
from .text import TextBiasDetector, TextBiasReport
from .image import ImageBiasDetector, ImageBiasReport
from .metrics import (
    demographic_parity,
    disparate_impact,
    equal_opportunity_difference,
    statistical_parity_difference,
    representation_ratio,
    sentiment_skew,
    stereotype_association,
    log_probability_bias_score,
    normalized_pointwise_mutual_information,
    bias_index,
)
from .mitigate import BiasMitigator
from .audit import BiasAuditor

__version__ = "0.1.0"
__all__ = [
    "TextBiasDetector",
    "TextBiasReport",
    "ImageBiasDetector",
    "ImageBiasReport",
    "BiasMitigator",
    "BiasAuditor",
    "demographic_parity",
    "disparate_impact",
    "equal_opportunity_difference",
    "statistical_parity_difference",
    "representation_ratio",
    "sentiment_skew",
    "stereotype_association",
    "log_probability_bias_score",
    "normalized_pointwise_mutual_information",
    "bias_index",
]
