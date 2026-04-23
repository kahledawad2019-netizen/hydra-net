"""
Per-Modality Explainability
===========================

Novelty contribution #5. Every decision from HYDRA-Net is accompanied by
an attribution trace showing *why* the system made the call.

Three levels of explanation, matching the cascade:

  1. Stage 1 exits: SHAP values over the handcrafted feature vector,
     grouped by modality (RF features vs audio features).

  2. Stage 2 exits: attention rollout across modalities — which tokens
     from which sensor mattered most.

  3. Stage 3 exits: per-drone SHAP on the GNN node embeddings + edge
     importance to show swarm-structure contributions.

Output format is designed to be readable by a human operator in a
dashboard (e.g., "Threat flagged because: RF 2.4 GHz band power anomaly
(62%) + propeller harmonic at 180 Hz (28%) + weak thermal signature (10%)").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttributionEntry:
    """One contribution to a decision."""
    source: str                   # e.g. "RF 2.4 GHz band power"
    modality: str                 # "rf", "audio", "rgb", "ir", "radar"
    contribution: float           # signed importance, positive = toward "drone"
    relative_weight: float        # fraction of total |contribution|


@dataclass
class DecisionExplanation:
    """Full decision trace."""
    stage_exited: int             # 1, 2, or 3
    top_contributions: list[AttributionEntry]
    summary: str                  # human-readable one-liner
    confidence: float


# Readable names for Stage 1 features (matches features.extract_combined_features)
STAGE1_FEATURE_NAMES = [
    # RF (first 10 features)
    ("rf", "RF spectral entropy"),
    ("rf", "RF peak frequency"),
    ("rf", "RF bandwidth"),
    ("rf", "RF total power"),
    ("rf", "RF PAPR"),
    ("rf", "Power in 2.4 GHz band"),
    ("rf", "Power in 5.8 GHz band"),
    ("rf", "Power in 433 MHz band"),
    ("rf", "Power in 868 MHz band"),
    ("rf", "Power in 915 MHz band"),
    # Audio (next 20 features: 7 scalar + 13 MFCC)
    ("audio", "Audio RMS energy"),
    ("audio", "Zero-crossing rate"),
    ("audio", "Spectral centroid"),
    ("audio", "Spectral rolloff"),
    ("audio", "Spectral flatness"),
    ("audio", "Propeller band power"),
    ("audio", "Propeller peak count"),
    *[("audio", f"MFCC proxy band {i}") for i in range(13)],
]


def explain_stage1(
    shap_values: np.ndarray,
    confidence: float,
    top_k: int = 5,
) -> DecisionExplanation:
    """
    Build a human-readable explanation from SHAP values on a Stage 1 prediction.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for each feature (shape: [n_features]).
    confidence : float
        Model confidence.
    top_k : int
        Number of top contributions to report.
    """
    abs_vals = np.abs(shap_values)
    total = abs_vals.sum() + 1e-12
    top_idx = np.argsort(-abs_vals)[:top_k]

    entries = []
    for idx in top_idx:
        if idx >= len(STAGE1_FEATURE_NAMES):
            modality, name = "unknown", f"feature_{idx}"
        else:
            modality, name = STAGE1_FEATURE_NAMES[idx]
        entries.append(
            AttributionEntry(
                source=name,
                modality=modality,
                contribution=float(shap_values[idx]),
                relative_weight=float(abs_vals[idx] / total),
            )
        )

    # Build summary string
    top = entries[0] if entries else None
    if top:
        direction = "toward drone" if top.contribution > 0 else "against drone"
        summary = (
            f"Stage 1 decision (conf={confidence:.2f}). "
            f"Primary driver: {top.source} ({top.modality.upper()}, {direction}, "
            f"{top.relative_weight * 100:.0f}% of weight)."
        )
    else:
        summary = f"Stage 1 decision (conf={confidence:.2f})."

    return DecisionExplanation(
        stage_exited=1,
        top_contributions=entries,
        summary=summary,
        confidence=confidence,
    )


def explain_stage2_attention(
    per_modality_attention: dict[str, float],
    confidence: float,
    drone_class: int,
) -> DecisionExplanation:
    """
    Build explanation from Stage 2 cross-modal attention weights.
    """
    total = sum(abs(v) for v in per_modality_attention.values()) + 1e-12
    entries = sorted(
        (
            AttributionEntry(
                source=f"Modality: {mod}",
                modality=mod,
                contribution=float(v),
                relative_weight=float(abs(v) / total),
            )
            for mod, v in per_modality_attention.items()
        ),
        key=lambda e: -e.relative_weight,
    )

    top_mod = entries[0].modality.upper() if entries else "unknown"
    summary = (
        f"Stage 2 decision: drone class {drone_class} (conf={confidence:.2f}). "
        f"Most informative modality: {top_mod}."
    )

    return DecisionExplanation(
        stage_exited=2,
        top_contributions=entries,
        summary=summary,
        confidence=confidence,
    )


def format_explanation_for_operator(explanation: DecisionExplanation) -> str:
    """Format explanation as operator-friendly multi-line text."""
    lines = [
        "=" * 60,
        f"DECISION (Stage {explanation.stage_exited} exit)",
        "=" * 60,
        explanation.summary,
        "",
        "Top contributing factors:",
    ]
    for i, entry in enumerate(explanation.top_contributions, 1):
        direction = "↑" if entry.contribution > 0 else "↓"
        lines.append(
            f"  {i}. [{entry.modality.upper():>5}] {entry.source:<35} "
            f"{direction} {entry.relative_weight * 100:5.1f}%"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
