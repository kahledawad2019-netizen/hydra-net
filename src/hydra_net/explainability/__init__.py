"""Per-modality explainability and SHAP-based attribution."""
from .attribution import (
    AttributionEntry,
    DecisionExplanation,
    STAGE1_FEATURE_NAMES,
    explain_stage1,
    explain_stage2_attention,
    format_explanation_for_operator,
)

__all__ = [
    "AttributionEntry",
    "DecisionExplanation",
    "STAGE1_FEATURE_NAMES",
    "explain_stage1",
    "explain_stage2_attention",
    "format_explanation_for_operator",
]
