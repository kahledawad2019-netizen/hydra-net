"""
Stage 1: Fast XGBoost Triage
=============================

The first stage of the HYDRA-Net cascade. Purpose: handle the ~90% of "easy"
inputs (empty sky, obvious birds, confident drone detections) in ~2 ms using
handcrafted features from RF and audio modalities.

If confidence >= CONFIDENCE_THRESHOLD, we emit the decision and skip
Stages 2 and 3 entirely. Otherwise, we escalate.

This stage is the primary source of the cascade's latency advantage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb


@dataclass
class Stage1Decision:
    """Output of Stage 1 triage."""
    label: int                # 0 = no drone, 1 = drone
    confidence: float         # [0, 1]
    should_exit: bool         # True if cascade should stop here
    inference_time_ms: float
    feature_vector: np.ndarray


class Stage1Triage:
    """
    Fast XGBoost-based binary triage classifier.

    Expects a handcrafted feature vector per sample. Features are computed
    by `features.py` from raw RF + audio streams and should be lightweight
    (no deep network inference).
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.95

    def __init__(
        self,
        model: Optional[xgb.XGBClassifier] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold

    @classmethod
    def new_untrained(cls, **xgb_kwargs) -> "Stage1Triage":
        """Create a fresh untrained classifier with sensible defaults."""
        defaults = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
        defaults.update(xgb_kwargs)
        return cls(model=xgb.XGBClassifier(**defaults))

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "Stage1Triage":
        """Train the triage classifier."""
        if self.model is None:
            raise RuntimeError("No model instantiated. Use Stage1Triage.new_untrained().")
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, x: np.ndarray) -> Stage1Decision:
        """
        Predict for a single sample and decide whether to exit the cascade.

        `x` is a 1D feature vector.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        x_batch = x.reshape(1, -1) if x.ndim == 1 else x
        start = time.perf_counter()
        proba = self.model.predict_proba(x_batch)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        label = int(np.argmax(proba))
        confidence = float(proba[label])
        should_exit = confidence >= self.confidence_threshold

        return Stage1Decision(
            label=label,
            confidence=confidence,
            should_exit=should_exit,
            inference_time_ms=elapsed_ms,
            feature_vector=x.flatten(),
        )

    def predict_batch(self, X: np.ndarray) -> list[Stage1Decision]:
        """Vectorized prediction for benchmarking. Per-sample decisions."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        start = time.perf_counter()
        probas = self.model.predict_proba(X)
        total_ms = (time.perf_counter() - start) * 1000
        per_sample_ms = total_ms / len(X)

        decisions = []
        for i, proba in enumerate(probas):
            label = int(np.argmax(proba))
            confidence = float(proba[label])
            decisions.append(
                Stage1Decision(
                    label=label,
                    confidence=confidence,
                    should_exit=confidence >= self.confidence_threshold,
                    inference_time_ms=per_sample_ms,
                    feature_vector=X[i],
                )
            )
        return decisions

    def save(self, path: str | Path) -> None:
        """Save trained model to JSON (XGBoost native format)."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> "Stage1Triage":
        """Load trained model from JSON."""
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return cls(model=model, confidence_threshold=confidence_threshold)
