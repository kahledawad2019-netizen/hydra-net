"""
HYDRA-Net Cascade Orchestrator
===============================

The top-level entry point. Routes a scene through the cascade:

  Stage 1 (XGBoost)           → exit if confident
    ↓ uncertain
  Stage 2 (Cross-modal Tx)    → exit if confident
    ↓ uncertain OR multi-target
  Stage 3 (GNN swarm reasoning)

Each stage is optional at inference time — the cascade gracefully handles
missing stages (e.g., Stage 1 only on Raspberry Pi deployment).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch

from .stage1 import Stage1Decision, Stage1Triage
from .stage2 import Stage2Decision, Stage2Module
from .stage3 import Stage3Decision, Stage3Module


@dataclass
class CascadeResult:
    """End-to-end result from a cascade inference."""
    final_stage: int
    total_latency_ms: float
    stage_latencies_ms: dict[int, float] = field(default_factory=dict)
    stage1_decision: Optional[Stage1Decision] = None
    stage2_decision: Optional[Stage2Decision] = None
    stage3_decision: Optional[Stage3Decision] = None
    cascade_path: list[str] = field(default_factory=list)


class HydraCascade:
    """
    Orchestrates the 3-stage cascade with confidence-gated early exit.

    Usage
    -----
        cascade = HydraCascade(stage1=s1, stage2=s2, stage3=s3)
        result = cascade.infer(
            stage1_features=x,           # required for Stage 1
            stage2_inputs={"rgb": ..., "ir": ..., "audio_spec": ...},
            stage3_inputs={"node_feats": ..., "positions": ..., "velocities": ...},
            force_full=False,            # if True, skip early exits
        )
    """

    def __init__(
        self,
        stage1: Optional[Stage1Triage] = None,
        stage2: Optional[Stage2Module] = None,
        stage3: Optional[Stage3Module] = None,
        multi_target_threshold: int = 2,
    ):
        if stage1 is None and stage2 is None and stage3 is None:
            raise ValueError("At least one stage must be provided.")
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.multi_target_threshold = multi_target_threshold

    def infer(
        self,
        stage1_features: Optional[np.ndarray] = None,
        stage2_inputs: Optional[dict] = None,
        stage3_inputs: Optional[dict] = None,
        force_full: bool = False,
    ) -> CascadeResult:
        total_start = time.perf_counter()
        result = CascadeResult(final_stage=0, total_latency_ms=0.0)

        # ----- Stage 1 -----
        if self.stage1 is not None and stage1_features is not None:
            d1 = self.stage1.predict(stage1_features)
            result.stage1_decision = d1
            result.stage_latencies_ms[1] = d1.inference_time_ms
            result.cascade_path.append(f"S1(conf={d1.confidence:.2f})")
            result.final_stage = 1

            if d1.should_exit and not force_full:
                result.total_latency_ms = (time.perf_counter() - total_start) * 1000
                result.cascade_path.append("EXIT")
                return result

        # ----- Stage 2 -----
        if self.stage2 is not None and stage2_inputs is not None:
            d2 = self.stage2.predict(**stage2_inputs)
            result.stage2_decision = d2
            result.stage_latencies_ms[2] = d2.inference_time_ms
            result.cascade_path.append(f"S2(cls={d2.drone_class},conf={d2.confidence:.2f})")
            result.final_stage = 2

            # Check if we should continue to Stage 3
            needs_swarm_reasoning = (
                stage3_inputs is not None
                and "node_feats" in stage3_inputs
                and stage3_inputs["node_feats"].size(0) >= self.multi_target_threshold
            )

            if d2.should_exit and not needs_swarm_reasoning and not force_full:
                result.total_latency_ms = (time.perf_counter() - total_start) * 1000
                result.cascade_path.append("EXIT")
                return result

        # ----- Stage 3 -----
        if self.stage3 is not None and stage3_inputs is not None:
            d3 = self.stage3.predict(**stage3_inputs)
            result.stage3_decision = d3
            result.stage_latencies_ms[3] = d3.inference_time_ms
            result.cascade_path.append(f"S3(n_drones={len(d3.per_drone)},threat={d3.scene_threat_level:.1f})")
            result.final_stage = 3

        result.total_latency_ms = (time.perf_counter() - total_start) * 1000
        return result


def summarize_cascade_result(result: CascadeResult) -> str:
    """Human-readable summary of a cascade result for debugging."""
    lines = [
        f"HYDRA-Net cascade: exited at Stage {result.final_stage}",
        f"Total latency: {result.total_latency_ms:.2f} ms",
        f"Path: {' → '.join(result.cascade_path)}",
    ]
    for stage, lat in sorted(result.stage_latencies_ms.items()):
        lines.append(f"  Stage {stage}: {lat:.2f} ms")
    if result.stage1_decision is not None:
        d = result.stage1_decision
        lines.append(f"  S1: label={d.label}, conf={d.confidence:.3f}, exit={d.should_exit}")
    if result.stage2_decision is not None:
        d = result.stage2_decision
        lines.append(f"  S2: class={d.drone_class}, conf={d.confidence:.3f}, exit={d.should_exit}")
    if result.stage3_decision is not None:
        d = result.stage3_decision
        lines.append(f"  S3: {len(d.per_drone)} drones, scene_threat={d.scene_threat_level:.2f}")
        for pd in d.per_drone[:3]:
            lines.append(
                f"    drone_{pd.drone_id}: threat={pd.threat_score:.2f}, "
                f"intent={pd.intent_class}, action={pd.recommended_action}"
            )
    return "\n".join(lines)
