"""
Asynchronous Multi-Rate Fusion + Meta-Learned Modality Gating
==============================================================

This module implements two core HYDRA-Net contributions:

  1. Async buffering: each sensor streams at its native rate; we buffer
     and align on demand when the cascade queries for a decision.

  2. Meta-learned modality gating: a tiny network that predicts, from
     environmental context (time-of-day, weather proxy from sensors,
     per-modality SNR), which modalities should be trusted *right now*.

This is fundamentally different from the "random modality dropout" in
existing literature — our gate is *context-aware*, not stochastic.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SensorFrame:
    """A single frame of data from one sensor."""
    modality: str                 # "rgb", "ir", "audio", "rf", "radar"
    data: np.ndarray | torch.Tensor
    timestamp_ns: int             # monotonic nanoseconds
    snr_estimate: float = 1.0     # sensor self-reported quality, [0, 1]


class AsyncSensorBuffer:
    """
    Ring buffer per sensor. Each sensor pushes frames as they arrive;
    the cascade pulls the most recent frame (or interpolates a time-aligned
    window) when it needs to make a decision.

    This decouples sensor rates from decision rate. A 60 fps RGB camera
    and a 10 Hz radar live happily in the same system.
    """

    def __init__(self, buffer_seconds: float = 2.0, expected_rates_hz: dict | None = None):
        self.buffer_seconds = buffer_seconds
        self.buffers: dict[str, deque] = {}
        self.expected_rates = expected_rates_hz or {}

    def push(self, frame: SensorFrame) -> None:
        if frame.modality not in self.buffers:
            # Estimate buffer size from rate (fall back to 1000 frames)
            rate = self.expected_rates.get(frame.modality, 100)
            maxlen = max(10, int(rate * self.buffer_seconds))
            self.buffers[frame.modality] = deque(maxlen=maxlen)
        self.buffers[frame.modality].append(frame)

    def latest(self, modality: str) -> SensorFrame | None:
        buf = self.buffers.get(modality)
        if not buf:
            return None
        return buf[-1]

    def snapshot(self, at_timestamp_ns: int | None = None) -> dict[str, SensorFrame | None]:
        """
        Return the closest-in-time frame from each modality.
        If at_timestamp_ns is None, use "now" (most recent per modality).
        """
        if at_timestamp_ns is None:
            return {mod: buf[-1] if buf else None for mod, buf in self.buffers.items()}

        snapshot = {}
        for mod, buf in self.buffers.items():
            if not buf:
                snapshot[mod] = None
                continue
            # Find closest timestamp
            closest = min(buf, key=lambda f: abs(f.timestamp_ns - at_timestamp_ns))
            # Only accept if within 100ms of requested time
            if abs(closest.timestamp_ns - at_timestamp_ns) < 100_000_000:
                snapshot[mod] = closest
            else:
                snapshot[mod] = None
        return snapshot

    def sensor_health(self) -> dict[str, dict]:
        """Report per-sensor health: rate, last-seen, mean SNR."""
        health = {}
        now_ns = time.monotonic_ns()
        for mod, buf in self.buffers.items():
            if not buf:
                health[mod] = {"healthy": False, "reason": "no data"}
                continue
            last = buf[-1]
            age_ms = (now_ns - last.timestamp_ns) / 1e6
            if len(buf) >= 2:
                dt_ns = buf[-1].timestamp_ns - buf[-2].timestamp_ns
                rate_hz = 1e9 / dt_ns if dt_ns > 0 else 0.0
            else:
                rate_hz = 0.0
            mean_snr = float(np.mean([f.snr_estimate for f in buf]))
            health[mod] = {
                "healthy": age_ms < 500,
                "age_ms": age_ms,
                "rate_hz": rate_hz,
                "mean_snr": mean_snr,
            }
        return health


class ModalityGate(nn.Module):
    """
    Meta-learned gate that predicts per-modality trust weights from context.

    Inputs (context vector):
      - per-modality SNR (how confident the sensor is in its own data)
      - per-modality age_ms (how stale is the last frame)
      - time-of-day sin/cos encoding (environmental prior)
      - ambient light proxy (from camera exposure or IR brightness)
      - estimated weather (from RF noise floor + IR gradient)

    Output:
      - softmax weights over modalities, summing to 1.
      - these weights modulate Stage 2 / Stage 3 fusion.
    """

    def __init__(self, n_modalities: int, context_dim: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.n_modalities = n_modalities
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_modalities),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, context_dim)
        returns: (B, n_modalities) — softmax weights
        """
        logits = self.net(context)
        return torch.softmax(logits, dim=-1)


def encode_time_of_day(hour: float) -> tuple[float, float]:
    """Cyclic encoding of hour-of-day in [0, 24)."""
    theta = 2 * np.pi * hour / 24.0
    return float(np.sin(theta)), float(np.cos(theta))


def build_context_vector(
    snrs: dict[str, float],
    ages_ms: dict[str, float],
    hour_of_day: float,
    ambient_light: float = 0.5,
    weather_proxy: float = 0.5,
    modality_order: tuple[str, ...] = ("rgb", "ir", "audio", "rf"),
) -> np.ndarray:
    """
    Assemble context vector consumed by ModalityGate.

    Parameters
    ----------
    snrs : dict
        Per-modality SNR in [0, 1].
    ages_ms : dict
        Per-modality age of last frame in ms.
    hour_of_day : float
        Local hour in [0, 24).
    ambient_light : float
        Proxy in [0, 1] — 0 = night, 1 = bright day.
    weather_proxy : float
        Proxy in [0, 1] — 1 = clear, 0 = heavy fog/rain.
    modality_order : tuple
        Ensures deterministic ordering.
    """
    snr_vec = [snrs.get(m, 0.0) for m in modality_order]
    age_vec = [1.0 / (1.0 + ages_ms.get(m, 1000) / 100) for m in modality_order]  # recency
    hour_sin, hour_cos = encode_time_of_day(hour_of_day)
    return np.array(
        snr_vec + age_vec + [hour_sin, hour_cos, ambient_light, weather_proxy],
        dtype=np.float32,
    )
