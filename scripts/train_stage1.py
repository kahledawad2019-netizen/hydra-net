"""
Train Stage 1 Triage Classifier
================================

Trains the XGBoost fast-triage on either synthetic data (PoC) or on
preprocessed features from real datasets (DroneRF, MPED-RF, etc.).

For real datasets, use the Colab notebook: notebooks/01_stage1_dronerf_colab.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from hydra_net.stage1 import Stage1Triage


def train_stage1(
    X_path: Path,
    y_path: Path,
    model_output: Path,
    results_output: Path,
    test_size: float = 0.2,
    seed: int = 42,
    confidence_threshold: float = 0.95,
) -> dict:
    """Train Stage 1 and report metrics."""
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    triage = Stage1Triage.new_untrained()
    triage.confidence_threshold = confidence_threshold

    print("\nTraining XGBoost...")
    triage.fit(X_train, y_train)

    # Evaluate
    decisions = triage.predict_batch(X_test)
    y_pred = np.array([d.label for d in decisions])
    y_proba = np.array([d.confidence if d.label == 1 else 1 - d.confidence for d in decisions])

    # Cascade-specific metrics
    exits_at_stage1 = np.array([d.should_exit for d in decisions])
    exit_rate = float(exits_at_stage1.mean())

    # Accuracy among exits (should be very high — this is what justifies the cascade)
    if exits_at_stage1.any():
        exit_accuracy = accuracy_score(y_test[exits_at_stage1], y_pred[exits_at_stage1])
    else:
        exit_accuracy = float("nan")

    metrics = {
        "overall_accuracy": float(accuracy_score(y_test, y_pred)),
        "overall_f1": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "overall_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "stage1_exit_rate": exit_rate,
        "stage1_exit_accuracy": float(exit_accuracy),
        "confidence_threshold": confidence_threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
    }

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING RESULTS")
    print("=" * 60)
    print(f"Overall accuracy:         {metrics['overall_accuracy']:.4f}")
    print(f"Overall F1:               {metrics['overall_f1']:.4f}")
    print(f"Overall ROC-AUC:          {metrics['overall_roc_auc']:.4f}")
    print(f"Stage 1 exit rate:        {metrics['stage1_exit_rate']:.2%}")
    print(f"Accuracy among exits:     {metrics['stage1_exit_accuracy']:.4f}")
    print(f"Confidence threshold:     {confidence_threshold}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save
    model_output.parent.mkdir(parents=True, exist_ok=True)
    triage.save(model_output)
    print(f"\nModel saved to {model_output}")

    results_output.parent.mkdir(parents=True, exist_ok=True)
    with open(results_output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {results_output}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--X-path", type=Path, default=None)
    parser.add_argument("--y-path", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=Path("models/stage1_triage.json"))
    parser.add_argument("--results-output", type=Path, default=Path("results/stage1_synthetic.json"))
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    args = parser.parse_args()

    if args.synthetic:
        X_path = args.X_path or Path("data/synthetic/X_synthetic.npy")
        y_path = args.y_path or Path("data/synthetic/y_synthetic.npy")
    else:
        if args.X_path is None or args.y_path is None:
            raise ValueError("Must provide --X-path and --y-path when not using --synthetic")
        X_path = args.X_path
        y_path = args.y_path

    train_stage1(
        X_path=X_path,
        y_path=y_path,
        model_output=args.model_output,
        results_output=args.results_output,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
