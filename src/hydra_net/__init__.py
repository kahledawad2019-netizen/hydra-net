"""
HYDRA-Net: Hierarchical Yield-Driven Resilient Async-fusion Network
====================================================================

A cascaded multimodal architecture for counter-UAV detection with
confidence-gated early exit, asynchronous multi-rate fusion,
meta-learned modality gating, threat+intent output, and per-modality
explainability.

Author: Khaled Metwalie

Note on imports: this package is designed so that Stage 1 (XGBoost) can
be used without PyTorch installed — critical for edge deployment
(Raspberry Pi, Jetson Nano). Torch-dependent components (cascade
orchestrator, Stage 2, Stage 3, fusion) are only imported when accessed.
"""

__version__ = "0.1.0-alpha"


def __getattr__(name: str):
    """Lazy import of torch-dependent components."""
    if name in ("CascadeResult", "HydraCascade", "summarize_cascade_result"):
        from . import cascade as _cascade
        return getattr(_cascade, name)
    raise AttributeError(f"module 'hydra_net' has no attribute {name!r}")


__all__ = ["CascadeResult", "HydraCascade", "summarize_cascade_result"]
