"""
HYDRA-Net Quickstart Demo
==========================

Runs the full PoC pipeline end-to-end in one command:
  1. Generate synthetic data
  2. Train Stage 1
  3. Run latency benchmark
  4. Show sample decisions with explanations

Usage:
  PYTHONPATH=src python scripts/quickstart_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def run(cmd, desc):
    print("\n" + "=" * 70)
    print(f"▶ {desc}")
    print("=" * 70)
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env={**__import__("os").environ, **env})
    if result.returncode != 0:
        print(f"⚠  Command failed: {' '.join(cmd)}")
        return False
    return True


def sample_explanations():
    """Show a few example decisions with SHAP-style explanations."""
    print("\n" + "=" * 70)
    print("▶ Sample decisions with explanations")
    print("=" * 70)

    from hydra_net.stage1 import Stage1Triage
    from hydra_net.explainability import explain_stage1, format_explanation_for_operator

    X = np.load(REPO_ROOT / "data/synthetic/X_synthetic.npy")
    y = np.load(REPO_ROOT / "data/synthetic/y_synthetic.npy")
    triage = Stage1Triage.load(REPO_ROOT / "models/stage1_triage.json")

    # Pick one drone and one non-drone sample
    drone_idx = int(np.where(y == 1)[0][0])
    no_drone_idx = int(np.where(y == 0)[0][0])

    # Compute approximate SHAP via feature-importance × feature-value. This is
    # a cheap proxy for the demo; it is NOT true SHAP. For publication-quality
    # attributions, use `shap.TreeExplainer(triage.model).shap_values(X)` — the
    # Colab notebooks do this. The proxy is sufficient to exercise the
    # explanation formatter and demonstrate the output schema.
    importances = triage.model.feature_importances_

    for idx, label in [(drone_idx, "DRONE"), (no_drone_idx, "NO DRONE")]:
        d = triage.predict(X[idx])
        shap_proxy = importances * X[idx]
        # Center around mean to get signed contributions
        shap_proxy = shap_proxy - shap_proxy.mean()
        exp = explain_stage1(shap_proxy, confidence=d.confidence, top_k=4)
        print(f"\n[Ground truth: {label}]  Predicted: {'DRONE' if d.label == 1 else 'NO DRONE'}")
        print(format_explanation_for_operator(exp))


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          HYDRA-Net Quickstart Demo (PoC on synthetic data)       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    steps = [
        ([sys.executable, "scripts/generate_synthetic_data.py", "--n-samples", "1000"],
         "Step 1/3: Generating synthetic multimodal data"),
        ([sys.executable, "scripts/train_stage1.py", "--synthetic"],
         "Step 2/3: Training Stage 1 XGBoost triage"),
        ([sys.executable, "scripts/benchmark_stage1_only.py"],
         "Step 3/3: Measuring cascade latency"),
    ]

    for cmd, desc in steps:
        if not run(cmd, desc):
            print("\n✗ Demo aborted.")
            sys.exit(1)

    sample_explanations()

    print("\n" + "=" * 70)
    print("✓ Demo complete.")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Review results/ for latency and accuracy JSONs")
    print("  • Run tests: python -m pytest tests/ -v")
    print("  • Train on real data: open notebooks/01_stage1_dronerf_colab.ipynb")
    print("  • Serve via API: uvicorn hydra_net.serving.api:app --port 8000")


if __name__ == "__main__":
    main()
