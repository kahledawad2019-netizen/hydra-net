"""
Stage 3: Deep Fusion + GNN for Swarm Reasoning
================================================

Invoked only for genuinely hard cases: swarms, adversarial drones, or
Stage 2 uncertainty. Uses a Graph Neural Network to reason about
inter-drone relationships (formation, coordinated movement, threat vectoring).

Nodes:  detected objects (each with per-modality embeddings from Stage 2)
Edges:  spatial proximity + velocity coherence
Output: per-node threat score, predicted intent class, recommended action

This is the slowest stage (~60 ms) but handles the ~1-5% of truly hard
inputs that require multi-target relational reasoning.

Training notebook: notebooks/03_stage3_swarm_colab.ipynb
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


INTENT_CLASSES = ["benign", "surveillance", "smuggling", "attack", "unknown"]
ACTION_CLASSES = ["monitor", "alert_operator", "track", "jam", "intercept"]


@dataclass
class PerDroneOutput:
    """Per-drone output from Stage 3."""
    drone_id: int
    threat_score: float           # [0, 5]
    intent_class: str
    intent_confidence: float
    recommended_action: str


@dataclass
class Stage3Decision:
    """Output of Stage 3 for a (possibly multi-drone) scene."""
    per_drone: list[PerDroneOutput]
    scene_threat_level: float     # aggregate
    inference_time_ms: float
    swarm_detected: bool


class SimpleGraphLayer(nn.Module):
    """
    Minimal message-passing layer. For full GNN, use torch-geometric's
    `GATv2Conv` or `TransformerConv` — this implementation stays framework-free
    for portability.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.msg = nn.Linear(dim * 2, dim)
        self.update = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        node_feats: (N, D)
        adj: (N, N) binary or weighted
        """
        n = node_feats.size(0)
        src = node_feats.unsqueeze(1).expand(n, n, -1)   # (N, N, D)
        dst = node_feats.unsqueeze(0).expand(n, n, -1)   # (N, N, D)
        pairs = torch.cat([src, dst], dim=-1)            # (N, N, 2D)

        messages = F.gelu(self.msg(pairs))                # (N, N, D)
        weighted = messages * adj.unsqueeze(-1)
        aggregated = weighted.sum(dim=1)                  # (N, D)

        combined = torch.cat([node_feats, aggregated], dim=-1)
        new_feats = self.update(combined)
        return self.norm(F.gelu(new_feats) + node_feats)  # residual


class SwarmReasoningNetwork(nn.Module):
    """
    GNN over detected drones. Each node is a drone with features:
      - embedding from Stage 2
      - kinematics (position, velocity)
      - per-modality confidence
    """

    def __init__(
        self,
        node_dim: int = 128 + 6 + 3,    # stage2 emb + pos/vel + confidences
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_intent_classes: int = len(INTENT_CLASSES),
        n_action_classes: int = len(ACTION_CLASSES),
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([SimpleGraphLayer(hidden_dim) for _ in range(n_layers)])

        # Per-node heads
        self.threat_head = nn.Linear(hidden_dim, 1)                  # regression 0-5
        self.intent_head = nn.Linear(hidden_dim, n_intent_classes)
        self.action_head = nn.Linear(hidden_dim, n_action_classes)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> dict:
        h = self.input_proj(node_feats)
        for layer in self.layers:
            h = layer(h, adj)

        threat = torch.sigmoid(self.threat_head(h).squeeze(-1)) * 5.0
        intent_logits = self.intent_head(h)
        action_logits = self.action_head(h)

        return {
            "threat": threat,
            "intent_logits": intent_logits,
            "action_logits": action_logits,
            "node_embeddings": h,
        }


def build_adjacency_from_kinematics(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    spatial_threshold: float = 50.0,
    velocity_coherence_weight: float = 0.5,
) -> torch.Tensor:
    """
    Build adjacency matrix from drone positions and velocities.

    Two drones are connected if:
      - spatially close (< spatial_threshold meters), and/or
      - moving with coherent velocity (coordinated swarm behavior)

    Returns a (N, N) weighted adjacency matrix in [0, 1].
    """
    n = positions.size(0)
    if n <= 1:
        return torch.ones(max(n, 1), max(n, 1))

    # Spatial proximity
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)       # (N, N, 3)
    dists = diffs.norm(dim=-1)                                    # (N, N)
    spatial = torch.exp(-dists / spatial_threshold)

    # Velocity coherence (cosine similarity)
    v_norm = F.normalize(velocities, dim=-1)
    vel_coh = (v_norm @ v_norm.t()).clamp(0, 1)                   # (N, N)

    adj = (1 - velocity_coherence_weight) * spatial + velocity_coherence_weight * vel_coh
    # Zero diagonal
    adj = adj * (1 - torch.eye(n, device=adj.device))
    return adj


class Stage3Module:
    """Inference wrapper for Stage 3."""

    def __init__(self, model: SwarmReasoningNetwork, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        velocities: torch.Tensor | None = None,
    ) -> Stage3Decision:
        """
        Predict per-drone threat + intent + action.

        Supply either `adj` directly or `positions` + `velocities` (will
        build adjacency from kinematics).
        """
        start = time.perf_counter()

        node_feats = node_feats.to(self.device)
        if adj is None:
            if positions is None or velocities is None:
                raise ValueError("Must provide either adj or (positions, velocities).")
            adj = build_adjacency_from_kinematics(positions, velocities).to(self.device)
        else:
            adj = adj.to(self.device)

        out = self.model(node_feats, adj)

        threats = out["threat"].cpu().numpy()
        intent_probs = F.softmax(out["intent_logits"], dim=-1).cpu().numpy()
        action_probs = F.softmax(out["action_logits"], dim=-1).cpu().numpy()

        per_drone = []
        for i in range(node_feats.size(0)):
            intent_idx = int(intent_probs[i].argmax())
            action_idx = int(action_probs[i].argmax())
            per_drone.append(
                PerDroneOutput(
                    drone_id=i,
                    threat_score=float(threats[i]),
                    intent_class=INTENT_CLASSES[intent_idx],
                    intent_confidence=float(intent_probs[i][intent_idx]),
                    recommended_action=ACTION_CLASSES[action_idx],
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return Stage3Decision(
            per_drone=per_drone,
            scene_threat_level=float(threats.max()) if len(threats) else 0.0,
            inference_time_ms=elapsed_ms,
            swarm_detected=node_feats.size(0) >= 3,
        )
