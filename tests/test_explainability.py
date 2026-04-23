"""Tests for explainability module (no torch required)."""
from __future__ import annotations

import numpy as np

from hydra_net.explainability import (
    STAGE1_FEATURE_NAMES,
    explain_stage1,
    explain_stage2_attention,
    format_explanation_for_operator,
)


def test_stage1_feature_names_length_matches_dim():
    # 30 features expected (10 RF + 20 audio)
    assert len(STAGE1_FEATURE_NAMES) == 30


def test_explain_stage1_basic():
    shap_vals = np.zeros(30, dtype=np.float32)
    shap_vals[5] = 0.8   # big positive (Power in 2.4 GHz band)
    shap_vals[15] = -0.3 # negative (Propeller band power)
    exp = explain_stage1(shap_vals, confidence=0.97, top_k=3)
    assert exp.stage_exited == 1
    assert len(exp.top_contributions) == 3
    assert exp.top_contributions[0].source == "Power in 2.4 GHz band"
    assert exp.top_contributions[0].modality == "rf"
    assert "2.4 GHz" in exp.summary


def test_explain_stage2_attention():
    per_mod = {"rgb": 0.6, "ir": 0.3, "audio": 0.1}
    exp = explain_stage2_attention(per_mod, confidence=0.9, drone_class=2)
    assert exp.stage_exited == 2
    assert exp.top_contributions[0].modality == "rgb"
    assert "class 2" in exp.summary or "class: 2" in exp.summary or "drone class 2" in exp.summary


def test_relative_weights_sum_to_one():
    shap_vals = np.random.randn(30).astype(np.float32)
    exp = explain_stage1(shap_vals, confidence=0.85, top_k=30)
    total = sum(e.relative_weight for e in exp.top_contributions)
    assert abs(total - 1.0) < 1e-5


def test_format_explanation_for_operator():
    shap_vals = np.zeros(30, dtype=np.float32)
    shap_vals[0] = 0.5
    exp = explain_stage1(shap_vals, confidence=0.9)
    out = format_explanation_for_operator(exp)
    assert "DECISION" in out
    assert "Stage 1" in out
    assert "RF" in out  # modality label present
