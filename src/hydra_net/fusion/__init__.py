"""Async multi-rate fusion and meta-learned modality gating."""
from .async_fusion import (
    AsyncSensorBuffer,
    ModalityGate,
    SensorFrame,
    build_context_vector,
    encode_time_of_day,
)

__all__ = [
    "AsyncSensorBuffer",
    "ModalityGate",
    "SensorFrame",
    "build_context_vector",
    "encode_time_of_day",
]
