"""Unit tests for Stage 1 triage."""
from __future__ import annotations

import numpy as np
import pytest

from hydra_net.stage1 import (
    FeatureConfig,
    Stage1Triage,
    extract_audio_features,
    extract_combined_features,
    extract_rf_features,
    feature_dim,
)


def test_feature_dim_matches_output():
    config = FeatureConfig()
    expected = feature_dim(config)
    rng = np.random.default_rng(0)
    rf = rng.normal(0, 1, 2000).astype(np.complex64)
    audio = rng.normal(0, 1, 8000).astype(np.float32)
    feats = extract_combined_features(rf, audio, config)
    assert feats.shape == (expected,)
    assert feats.dtype == np.float32


def test_rf_features_finite():
    rng = np.random.default_rng(1)
    rf = rng.normal(0, 1, 1000).astype(np.complex64)
    feats = extract_rf_features(rf, FeatureConfig())
    assert np.all(np.isfinite(feats))


def test_audio_features_finite():
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 1, 8000).astype(np.float32)
    feats = extract_audio_features(audio, FeatureConfig())
    assert np.all(np.isfinite(feats))


def test_audio_features_handle_silence():
    audio = np.zeros(8000, dtype=np.float32)
    feats = extract_audio_features(audio, FeatureConfig())
    assert np.all(np.isfinite(feats))


def test_triage_predict_returns_decision():
    triage = Stage1Triage.new_untrained()
    # Tiny synthetic dataset
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (60, 30)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    triage.fit(X, y)
    d = triage.predict(X[0])
    assert d.label in (0, 1)
    assert 0.0 <= d.confidence <= 1.0
    assert d.inference_time_ms >= 0
    assert isinstance(d.should_exit, bool)


def test_triage_save_load_roundtrip(tmp_path):
    triage = Stage1Triage.new_untrained()
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (50, 30)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    triage.fit(X, y)
    path = tmp_path / "stage1.json"
    triage.save(path)
    reloaded = Stage1Triage.load(path, confidence_threshold=0.8)
    d1 = triage.predict(X[0])
    d2 = reloaded.predict(X[0])
    assert d1.label == d2.label
    assert abs(d1.confidence - d2.confidence) < 1e-5


def test_triage_batch_predict():
    triage = Stage1Triage.new_untrained()
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (40, 30)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    triage.fit(X, y)
    decisions = triage.predict_batch(X)
    assert len(decisions) == len(X)
    assert all(d.label in (0, 1) for d in decisions)


def test_triage_raises_without_fit():
    triage = Stage1Triage.new_untrained()
    with pytest.raises(Exception):
        triage.predict(np.zeros(30, dtype=np.float32))


def test_confidence_threshold_controls_exit():
    triage = Stage1Triage.new_untrained()
    rng = np.random.default_rng(6)
    # Make data very separable so confidence is high
    X = np.concatenate([
        rng.normal(-5, 0.5, (50, 30)),
        rng.normal(5, 0.5, (50, 30)),
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int32)
    triage.fit(X, y)

    # Very high threshold → fewer exits
    triage.confidence_threshold = 0.999999
    strict = sum(triage.predict(x).should_exit for x in X)
    # Lenient threshold → more exits
    triage.confidence_threshold = 0.5
    lenient = sum(triage.predict(x).should_exit for x in X)
    assert lenient >= strict
