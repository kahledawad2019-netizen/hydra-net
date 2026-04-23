"""
Synthetic Data Generator for HYDRA-Net PoC
===========================================

Generates physically-plausible RF and audio signals for both drone-present
and drone-absent classes, so Stage 1 can be trained and benchmarked
without downloading the full ~150 GB of real datasets.

This is ONLY for proof-of-concept and latency benchmarking. Real training
must use DroneRF, MPED-RF, etc. (see notebooks/).

Synthetic classes:
  0: empty sky (ambient noise, weak thermal/wind noise only)
  1: quadcopter drone (2.4 GHz control + propeller harmonics ~150-250 Hz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hydra_net.stage1 import FeatureConfig, extract_combined_features


def generate_rf_signal(
    has_drone: bool,
    sample_rate: float = 20e6,
    duration_s: float = 0.01,
    noise_floor: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a short RF signal snippet.

    Drone-present: adds narrowband bursts near 2.4 GHz-like frequencies
                   (after downconversion to baseband these show as peaks
                   in the spectrum).
    Drone-absent: pure Gaussian noise floor with occasional WiFi-like bursts.
    """
    rng = rng or np.random.default_rng()
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate

    # Baseline noise
    signal = rng.normal(0, noise_floor, n_samples).astype(np.complex64)
    signal = signal + 1j * rng.normal(0, noise_floor, n_samples).astype(np.float32)

    if has_drone:
        # Add 2-3 narrowband tones (simulating frequency-hopping control channel
        # after downconversion)
        n_tones = rng.integers(2, 4)
        for _ in range(n_tones):
            # After downconversion, hopping channel appears at various offsets
            freq_offset = rng.uniform(-5e6, 5e6)
            amplitude = rng.uniform(0.5, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            # Short burst (not continuous — FHSS pattern)
            burst_start = rng.integers(0, n_samples // 2)
            burst_len = rng.integers(n_samples // 10, n_samples // 3)
            burst_end = min(burst_start + burst_len, n_samples)
            burst_t = t[burst_start:burst_end]
            tone = amplitude * np.exp(1j * (2 * np.pi * freq_offset * burst_t + phase))
            signal[burst_start:burst_end] += tone.astype(np.complex64)
    else:
        # Occasional WiFi-like spurious burst (to make the classification non-trivial)
        if rng.random() < 0.2:
            freq_offset = rng.uniform(-8e6, 8e6)
            amplitude = rng.uniform(0.2, 0.4)
            tone = amplitude * np.exp(1j * 2 * np.pi * freq_offset * t)
            signal += tone.astype(np.complex64)

    return signal


def generate_audio_signal(
    has_drone: bool,
    sample_rate: int = 16000,
    duration_s: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a short audio clip.

    Drone-present: propeller harmonics at 150-250 Hz fundamental with
                   multiple harmonics, slightly modulated (quadcopter has
                   4 props at slightly different RPMs).
    Drone-absent: ambient noise, possibly with bird / wind.
    """
    rng = rng or np.random.default_rng()
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate

    # Ambient background
    signal = rng.normal(0, 0.05, n_samples).astype(np.float32)

    if has_drone:
        # 4 propellers at slightly different RPMs
        n_props = 4
        for _ in range(n_props):
            fundamental = rng.uniform(150, 250)
            # Propeller creates fundamental + several harmonics
            n_harmonics = rng.integers(3, 6)
            amplitude = rng.uniform(0.1, 0.3)
            phase = rng.uniform(0, 2 * np.pi)
            for h in range(1, n_harmonics + 1):
                harm_amp = amplitude / h     # decreasing amplitude
                signal += harm_amp * np.sin(2 * np.pi * fundamental * h * t + phase)

        # Add some amplitude modulation (rotor blade passage)
        am_freq = rng.uniform(15, 30)
        am_depth = rng.uniform(0.1, 0.3)
        signal *= 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
    else:
        # Maybe bird call (tonal) or wind (low-frequency rumble)
        if rng.random() < 0.3:
            # Bird-like: short chirp
            chirp_freq_start = rng.uniform(2000, 4000)
            chirp_freq_end = rng.uniform(3000, 6000)
            chirp_duration = rng.uniform(0.05, 0.15)
            chirp_start = rng.uniform(0, duration_s - chirp_duration)
            chirp_mask = (t >= chirp_start) & (t < chirp_start + chirp_duration)
            chirp_t = t[chirp_mask] - chirp_start
            chirp_freq = np.linspace(chirp_freq_start, chirp_freq_end, chirp_t.size)
            signal[chirp_mask] += 0.3 * np.sin(2 * np.pi * chirp_freq * chirp_t)
        elif rng.random() < 0.5:
            # Wind rumble: low-frequency filtered noise
            rumble = rng.normal(0, 0.1, n_samples)
            # Simple LP filter
            rumble = np.convolve(rumble, np.ones(200) / 200, mode="same")
            signal += rumble.astype(np.float32)

    return signal


def generate_dataset(
    n_samples: int = 2000,
    drone_ratio: float = 0.5,
    seed: int = 42,
    output_dir: Path = Path("data/synthetic"),
) -> dict:
    """
    Generate a synthetic multimodal dataset.

    Returns dict with X (feature matrix), y (labels), and metadata.
    Also saves to disk for reuse.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    config = FeatureConfig()
    labels = rng.random(n_samples) < drone_ratio
    X = []
    y = []

    for has_drone in tqdm(labels, desc="Generating synthetic data"):
        rf = generate_rf_signal(has_drone=bool(has_drone), rng=rng)
        audio = generate_audio_signal(has_drone=bool(has_drone), rng=rng)
        feats = extract_combined_features(rf, audio, config)
        X.append(feats)
        y.append(int(has_drone))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.save(output_dir / "X_synthetic.npy", X)
    np.save(output_dir / "y_synthetic.npy", y)

    metadata = {
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "class_balance": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
        "seed": seed,
    }
    print(f"\nGenerated {n_samples} samples, {X.shape[1]} features each.")
    print(f"Class balance: {metadata['class_balance']}")
    print(f"Saved to {output_dir}/")

    return {"X": X, "y": y, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic HYDRA-Net PoC dataset")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--drone-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        drone_ratio=args.drone_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
