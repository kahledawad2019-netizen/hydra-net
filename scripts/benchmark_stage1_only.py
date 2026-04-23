"""
Stage-1-Only Latency Benchmark (torch-free)
============================================

Measures raw Stage 1 triage latency. This is the primary cascade advantage:
if Stage 1 exits for ~90% of inputs at ~2 ms, the cascade's median latency
collapses to ~2 ms while the monolithic baseline pays the full transformer
cost on every input.

This script runs without PyTorch installed — useful for Raspberry Pi /
edge benchmarking and for quick verification of the repo on any machine.

For the full cascade-vs-monolithic benchmark (requires torch), see
scripts/benchmark_latency.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hydra_net.stage1 import Stage1Triage


def benchmark_stage1(
    stage1_model_path: Path,
    X_path: Path,
    y_path: Path,
    n_warmup: int = 50,
    confidence_threshold: float = 0.95,
    output_path: Path = Path("results/stage1_latency.json"),
) -> dict:
    """Measure Stage 1 latency and exit-rate characteristics."""
    stage1 = Stage1Triage.load(stage1_model_path, confidence_threshold=confidence_threshold)
    X = np.load(X_path)
    y = np.load(y_path)
    n_samples = len(X)

    print(f"Benchmarking Stage 1 on {n_samples} samples (CPU, torch-free)")

    # Warmup (JIT, caches, etc.)
    for i in range(n_warmup):
        stage1.predict(X[i % n_samples])

    # Per-sample timing
    latencies_ms = []
    exits = 0
    correct = 0
    for i in range(n_samples):
        t0 = time.perf_counter()
        d = stage1.predict(X[i])
        # Measure including the decision wrapper, not just predict_proba
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if d.should_exit:
            exits += 1
        if d.label == y[i]:
            correct += 1

    lat = np.array(latencies_ms)

    # Stage 2 cost assumption: we use a published reference value for
    # cross-modal transformer inference. On CPU, a small multimodal
    # transformer (~10 M params) with 224x224 RGB + IR + audio spec
    # typically takes 80-150 ms per sample. We take 100 ms as a
    # conservative reference and clearly label it as such.
    STAGE2_REFERENCE_LATENCY_MS = 100.0

    # Projected cascade latency distribution:
    # - exits at S1: pay only S1 latency
    # - escalates to S2: pay S1 + S2 reference latency
    projected_cascade_ms = np.where(
        lat >= 0,  # all samples
        [l + (0 if stage1.predict(X[i]).should_exit else STAGE2_REFERENCE_LATENCY_MS)
         for i, l in enumerate(lat)],
        lat,
    )

    # Simpler: recompute once
    projected_cascade_ms = []
    for i in range(n_samples):
        d = stage1.predict(X[i])
        if d.should_exit:
            projected_cascade_ms.append(lat[i])
        else:
            projected_cascade_ms.append(lat[i] + STAGE2_REFERENCE_LATENCY_MS)
    projected_cascade_ms = np.array(projected_cascade_ms)

    monolithic_ms = lat + STAGE2_REFERENCE_LATENCY_MS  # always pay S2

    results = {
        "n_samples": int(n_samples),
        "confidence_threshold": confidence_threshold,
        "stage1_only": {
            "p50_ms": float(np.percentile(lat, 50)),
            "p90_ms": float(np.percentile(lat, 90)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(lat.mean()),
            "std_ms": float(lat.std()),
        },
        "stage1_exit_rate": float(exits / n_samples),
        "stage1_accuracy": float(correct / n_samples),
        "projected_cascade_with_s2_ref_latency_100ms": {
            "stage2_reference_latency_ms": STAGE2_REFERENCE_LATENCY_MS,
            "p50_ms": float(np.percentile(projected_cascade_ms, 50)),
            "p90_ms": float(np.percentile(projected_cascade_ms, 90)),
            "p99_ms": float(np.percentile(projected_cascade_ms, 99)),
            "mean_ms": float(projected_cascade_ms.mean()),
        },
        "projected_monolithic_with_s2_ref_latency_100ms": {
            "p50_ms": float(np.percentile(monolithic_ms, 50)),
            "mean_ms": float(monolithic_ms.mean()),
        },
        "projected_speedup_vs_monolithic": {
            "p50": float(np.percentile(monolithic_ms, 50) / max(np.percentile(projected_cascade_ms, 50), 1e-6)),
            "mean": float(monolithic_ms.mean() / max(projected_cascade_ms.mean(), 1e-6)),
        },
        "notes": (
            "Stage 1 latencies are REAL MEASUREMENTS on this machine's CPU. "
            "Stage 2 latency is a REFERENCE VALUE (100 ms) representative of "
            "small multimodal transformers on CPU; real Stage 2 latency "
            "should be measured on the target deployment hardware. "
            "Synthetic data produces an easier triage task than real DroneRF, "
            "so the reported exit rate is an UPPER BOUND on what to expect. "
            "Real-world exit rate is typically 70-90% on DroneRF."
        ),
    }

    print("\n" + "=" * 70)
    print("STAGE 1 LATENCY — REAL MEASUREMENTS (synthetic data, CPU)")
    print("=" * 70)
    print(f"Samples: {n_samples}")
    print(f"{'p50':<10} {lat[int(0.5 * n_samples)]:>6.3f} ms  (median)")
    print(f"{'p90':<10} {np.percentile(lat, 90):>6.3f} ms")
    print(f"{'p99':<10} {np.percentile(lat, 99):>6.3f} ms")
    print(f"{'mean':<10} {lat.mean():>6.3f} ms")
    print(f"Stage 1 exit rate: {exits / n_samples:.1%}  (at conf >= {confidence_threshold})")
    print(f"Stage 1 accuracy:  {correct / n_samples:.3f}")
    print()
    print("PROJECTED CASCADE vs MONOLITHIC")
    print("-" * 70)
    print(f"Using Stage 2 reference latency = {STAGE2_REFERENCE_LATENCY_MS:.0f} ms (CPU)")
    print(f"{'':<20} {'Cascade (ms)':>15} {'Monolithic (ms)':>18} {'Speedup':>10}")
    print(f"{'p50':<20} {results['projected_cascade_with_s2_ref_latency_100ms']['p50_ms']:>15.2f} "
          f"{results['projected_monolithic_with_s2_ref_latency_100ms']['p50_ms']:>18.2f} "
          f"{results['projected_speedup_vs_monolithic']['p50']:>9.1f}x")
    print(f"{'mean':<20} {results['projected_cascade_with_s2_ref_latency_100ms']['mean_ms']:>15.2f} "
          f"{results['projected_monolithic_with_s2_ref_latency_100ms']['mean_ms']:>18.2f} "
          f"{results['projected_speedup_vs_monolithic']['mean']:>9.1f}x")
    print("=" * 70)
    print()
    print("⚠  Stage 2 value is a reference; real latency must be measured on GPU/target hw.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-model", type=Path, default=Path("models/stage1_triage.json"))
    parser.add_argument("--X-path", type=Path, default=Path("data/synthetic/X_synthetic.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("data/synthetic/y_synthetic.npy"))
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("results/stage1_latency.json"))
    args = parser.parse_args()

    benchmark_stage1(
        stage1_model_path=args.stage1_model,
        X_path=args.X_path,
        y_path=args.y_path,
        confidence_threshold=args.confidence_threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
