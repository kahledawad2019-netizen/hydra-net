"""
Latency Benchmark: Cascade vs. Monolithic
==========================================

This is the single most important demonstration of HYDRA-Net's value.
We compare two inference strategies on the same test set:

  A) HYDRA-Net cascade: Stage 1 exits when confident; only ~10% of inputs
     proceed to the expensive Stage 2.

  B) Monolithic baseline: Stage 2 runs on every input (simulating the
     behavior of current SOTA multimodal transformers).

We measure: per-sample latency distribution (p50, p90, p99), total wall
time, and Stage 2 invocation rate.

All numbers are REAL MEASUREMENTS from actual code, not fabricated SOTA
claims. When reporting results, we label them as synthetic-PoC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hydra_net import HydraCascade
from hydra_net.stage1 import Stage1Triage
from hydra_net.stage2 import CrossModalTransformer, Stage2Module


def make_fake_stage2_inputs(batch_size: int = 1, img_size: int = 224):
    """Generate random tensors matching expected Stage 2 input shapes."""
    return {
        "rgb": torch.randn(3, img_size, img_size),
        "ir": torch.randn(1, img_size, img_size),
        "audio_spec": torch.randn(1, 128, 128),
    }


def benchmark(
    stage1_model_path: Path,
    X_path: Path,
    y_path: Path,
    n_warmup: int = 20,
    n_benchmark: int | None = None,
    confidence_threshold: float = 0.95,
    device: str = "cpu",
    output_path: Path = Path("results/latency_benchmark.json"),
) -> dict:
    """Run latency benchmark comparing cascade vs. monolithic."""

    # Load Stage 1
    stage1 = Stage1Triage.load(stage1_model_path, confidence_threshold=confidence_threshold)

    # Build Stage 2 (untrained — latency is the same whether trained or not)
    stage2_model = CrossModalTransformer()
    stage2 = Stage2Module(stage2_model, device=device)

    # Load test data
    X = np.load(X_path)
    y = np.load(y_path)
    if n_benchmark is not None:
        X = X[:n_benchmark]
        y = y[:n_benchmark]
    n_samples = len(X)

    print(f"Benchmarking on {n_samples} samples (device={device})...")

    # ------ Warmup ------
    print(f"Warming up ({n_warmup} iterations)...")
    for i in range(n_warmup):
        idx = i % n_samples
        stage1.predict(X[idx])
        stage2.predict(**make_fake_stage2_inputs())

    # ------ Cascade run ------
    print("\n[Strategy A] HYDRA-Net cascade (early exit enabled)")
    cascade = HydraCascade(stage1=stage1, stage2=stage2)
    cascade_latencies = []
    cascade_exits_at_s1 = 0
    wall_start = time.perf_counter()
    for i in range(n_samples):
        result = cascade.infer(
            stage1_features=X[i],
            stage2_inputs=make_fake_stage2_inputs(),
        )
        cascade_latencies.append(result.total_latency_ms)
        if result.final_stage == 1:
            cascade_exits_at_s1 += 1
    cascade_wall_ms = (time.perf_counter() - wall_start) * 1000

    # ------ Monolithic run ------
    print("\n[Strategy B] Monolithic baseline (always run Stage 2)")
    monolithic_latencies = []
    wall_start = time.perf_counter()
    for i in range(n_samples):
        result = cascade.infer(
            stage1_features=X[i],
            stage2_inputs=make_fake_stage2_inputs(),
            force_full=True,
        )
        monolithic_latencies.append(result.total_latency_ms)
    monolithic_wall_ms = (time.perf_counter() - wall_start) * 1000

    cascade_lat = np.array(cascade_latencies)
    mono_lat = np.array(monolithic_latencies)

    results = {
        "n_samples": int(n_samples),
        "device": device,
        "confidence_threshold": confidence_threshold,
        "cascade": {
            "p50_ms": float(np.percentile(cascade_lat, 50)),
            "p90_ms": float(np.percentile(cascade_lat, 90)),
            "p99_ms": float(np.percentile(cascade_lat, 99)),
            "mean_ms": float(cascade_lat.mean()),
            "std_ms": float(cascade_lat.std()),
            "total_wall_ms": float(cascade_wall_ms),
            "stage1_exit_rate": float(cascade_exits_at_s1 / n_samples),
        },
        "monolithic": {
            "p50_ms": float(np.percentile(mono_lat, 50)),
            "p90_ms": float(np.percentile(mono_lat, 90)),
            "p99_ms": float(np.percentile(mono_lat, 99)),
            "mean_ms": float(mono_lat.mean()),
            "std_ms": float(mono_lat.std()),
            "total_wall_ms": float(monolithic_wall_ms),
        },
        "speedup": {
            "p50": float(np.percentile(mono_lat, 50) / max(np.percentile(cascade_lat, 50), 1e-6)),
            "mean": float(mono_lat.mean() / max(cascade_lat.mean(), 1e-6)),
            "wall_clock": float(monolithic_wall_ms / max(cascade_wall_ms, 1e-6)),
        },
        "note": "Synthetic PoC benchmark. Stage 2 is an untrained transformer — "
                "inference latency is representative of a trained model's compute "
                "cost. Real-dataset validation pending GPU training.",
    }

    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK RESULTS (Synthetic PoC, CPU)")
    print("=" * 70)
    print(f"Samples: {n_samples} | Confidence threshold: {confidence_threshold}")
    print()
    print(f"{'Metric':<25} {'Cascade':>15} {'Monolithic':>15} {'Speedup':>12}")
    print("-" * 70)
    print(f"{'p50 latency (ms)':<25} {results['cascade']['p50_ms']:>15.2f} "
          f"{results['monolithic']['p50_ms']:>15.2f} "
          f"{results['speedup']['p50']:>11.1f}x")
    print(f"{'p90 latency (ms)':<25} {results['cascade']['p90_ms']:>15.2f} "
          f"{results['monolithic']['p90_ms']:>15.2f}")
    print(f"{'p99 latency (ms)':<25} {results['cascade']['p99_ms']:>15.2f} "
          f"{results['monolithic']['p99_ms']:>15.2f}")
    print(f"{'Mean latency (ms)':<25} {results['cascade']['mean_ms']:>15.2f} "
          f"{results['monolithic']['mean_ms']:>15.2f} "
          f"{results['speedup']['mean']:>11.1f}x")
    print(f"{'Total wall time (ms)':<25} {results['cascade']['total_wall_ms']:>15.2f} "
          f"{results['monolithic']['total_wall_ms']:>15.2f} "
          f"{results['speedup']['wall_clock']:>11.1f}x")
    print()
    print(f"Stage 1 exit rate: {results['cascade']['stage1_exit_rate']:.1%}")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-model", type=Path, default=Path("models/stage1_triage.json"))
    parser.add_argument("--X-path", type=Path, default=Path("data/synthetic/X_synthetic.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("data/synthetic/y_synthetic.npy"))
    parser.add_argument("--n-samples", type=int, default=None, help="Limit benchmark sample count")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/latency_benchmark.json"))
    args = parser.parse_args()

    benchmark(
        stage1_model_path=args.stage1_model,
        X_path=args.X_path,
        y_path=args.y_path,
        n_warmup=args.n_warmup,
        n_benchmark=args.n_samples,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
