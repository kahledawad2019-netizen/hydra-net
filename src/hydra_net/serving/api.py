"""
HYDRA-Net FastAPI Serving Layer
================================

Exposes the cascade as an HTTP endpoint for integration with dashboards
(Power BI, Grafana), security operations centers, or other C-UAV systems.

Endpoints:
  GET  /health                 — readiness check + model status
  POST /predict/stage1         — Stage 1 only (edge deployment)
  POST /predict/cascade        — full cascade
  GET  /sensor-health          — per-sensor buffer status
  GET  /metrics                — Prometheus-compatible metrics

Run:
  uvicorn hydra_net.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hydra_net.stage1 import Stage1Triage


# ---------- Config ----------

DEFAULT_STAGE1_MODEL_PATH = Path("models/stage1_triage.json")


# ---------- Request / response schemas ----------

class Stage1Request(BaseModel):
    features: list[float] = Field(
        ...,
        description="30-d handcrafted feature vector (see hydra_net.stage1.features)",
        min_length=1,
    )


class Stage1Response(BaseModel):
    label: int
    confidence: float
    should_exit: bool
    inference_time_ms: float
    recommended_action: str


class HealthResponse(BaseModel):
    status: str
    stage1_loaded: bool
    stage2_loaded: bool
    stage3_loaded: bool
    version: str


# ---------- App + state ----------

app = FastAPI(
    title="HYDRA-Net Counter-UAV API",
    description="Cascaded multimodal anti-drone detection with confidence-gated early exit.",
    version="0.1.0-alpha",
)

# Simple in-process state (for production, use lifespan context manager)
_stage1: Optional[Stage1Triage] = None
_request_count = 0
_exit_count_s1 = 0
_total_latency_ms = 0.0


def _load_stage1():
    global _stage1
    if _stage1 is None and DEFAULT_STAGE1_MODEL_PATH.exists():
        _stage1 = Stage1Triage.load(DEFAULT_STAGE1_MODEL_PATH)


@app.on_event("startup")
def _startup():
    _load_stage1()


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        stage1_loaded=_stage1 is not None,
        stage2_loaded=False,  # wire up when Stage 2 checkpoint exists
        stage3_loaded=False,
        version="0.1.0-alpha",
    )


@app.post("/predict/stage1", response_model=Stage1Response)
def predict_stage1(req: Stage1Request):
    global _request_count, _exit_count_s1, _total_latency_ms

    if _stage1 is None:
        raise HTTPException(
            status_code=503,
            detail=f"Stage 1 model not loaded. Train it first and place at {DEFAULT_STAGE1_MODEL_PATH}",
        )

    x = np.asarray(req.features, dtype=np.float32)
    d = _stage1.predict(x)

    _request_count += 1
    if d.should_exit:
        _exit_count_s1 += 1
    _total_latency_ms += d.inference_time_ms

    # Simple recommended action mapping (in production, Stage 3 produces this)
    if d.label == 0:
        recommended = "monitor"
    elif d.confidence > 0.98:
        recommended = "alert_operator"
    else:
        recommended = "escalate_to_stage2"

    return Stage1Response(
        label=d.label,
        confidence=d.confidence,
        should_exit=d.should_exit,
        inference_time_ms=d.inference_time_ms,
        recommended_action=recommended,
    )


@app.get("/metrics")
def metrics():
    """Prometheus-compatible plain-text metrics."""
    avg_lat = _total_latency_ms / max(_request_count, 1)
    lines = [
        "# HELP hydra_requests_total Total inference requests processed",
        "# TYPE hydra_requests_total counter",
        f"hydra_requests_total {_request_count}",
        "# HELP hydra_stage1_exits_total Requests that exited at Stage 1",
        "# TYPE hydra_stage1_exits_total counter",
        f"hydra_stage1_exits_total {_exit_count_s1}",
        "# HELP hydra_average_latency_ms Average per-request latency in ms",
        "# TYPE hydra_average_latency_ms gauge",
        f"hydra_average_latency_ms {avg_lat:.4f}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
