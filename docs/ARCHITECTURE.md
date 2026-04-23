# HYDRA-Net Architecture Details

This document is a technical reference for the architecture. For the research
paper framing, see `PAPER_OUTLINE.md`. For user-facing documentation, see
the top-level `README.md`.

---

## The cascade in one page

```
                   Sensor streams (async)
          ┌──────────┬──────────┬──────────┐
          │          │          │          │
         RF       Audio      RGB+IR     Radar
       (20 MHz) (16 kHz)   (30-60 fps)  (10 Hz)
          │          │          │          │
          └──────────┴──────────┴──────────┘
                          │
                   AsyncSensorBuffer
                   (per-modality ring buffers,
                    native-rate)
                          │
        ┌─────────────────▼─────────────────┐
        │   Stage 1: XGBoost Triage (~2 ms) │
        │   Features: 30-d handcrafted       │
        │   Output: binary + confidence      │
        └─────────┬─────────────────┬────────┘
                  │ conf ≥ 0.95     │ conf < 0.95
                  ▼                 ▼
               ╔═════╗     ┌─────────────────────────┐
               ║EXIT ║     │  ModalityGate           │
               ╚═════╝     │  (context-aware)        │
                           └───────────┬─────────────┘
                                       ▼
                   ┌───────────────────────────────────┐
                   │   Stage 2: Cross-Modal            │
                   │   Transformer (~15 ms on GPU)     │
                   │   Inputs: RGB + IR + audio-spec   │
                   │   Output: drone class + conf      │
                   └────────┬─────────────────┬────────┘
                            │ conf ≥ 0.85 AND │ uncertain OR
                            │ single target   │ multi-target
                            ▼                 ▼
                         ╔═════╗     ┌───────────────────┐
                         ║EXIT ║     │  Stage 3: GNN     │
                         ╚═════╝     │  Swarm Reasoning  │
                                     │  (~60 ms)         │
                                     └─────────┬─────────┘
                                               ▼
                                  Per-drone:
                                    threat score (0-5)
                                    intent class
                                    recommended action
```

---

## Stage 1 in detail

### Feature vector (30 dimensions)

**RF features (10 dims):**
1-5. Scalar: spectral entropy, peak frequency, -3 dB bandwidth, total power, PAPR
6-10. Relative power in 5 known drone bands: 2.4 GHz ISM, 5.8 GHz ISM, 433/868/915 MHz

**Audio features (20 dims):**
11-17. Scalar: RMS, ZCR, spectral centroid, rolloff, flatness, propeller-band power, harmonic-peak count
18-30. MFCC proxy (13 log-mel bands)

### Why handcrafted over learned?

- **Latency:** no neural net forward pass; just scipy Welch PSD + FFT
- **Interpretability:** each feature is physically meaningful → SHAP explanations are human-readable
- **Sample efficiency:** XGBoost on 30 features trains well with 1000s of samples; a deep net needs 100x more

### Confidence calibration

Out-of-the-box XGBoost probabilities are not calibrated. We apply Platt scaling
on the validation set before deployment. The confidence threshold τ₁ = 0.95 is
Pareto-optimal on DroneRF (see paper Table 3).

### What Stage 1 handles well
- Strong RF signatures (2.4 GHz FHSS bursts)
- Clear propeller harmonics
- Quiet background (no drone + no clutter)

### What Stage 1 handles poorly (→ escalation)
- Weak RF (distant drone or RF-silent)
- Acoustic-only at distance (propeller sound masked)
- Bird vs drone at boundary of training distribution

---

## Stage 2 in detail

### Architecture

- **Patch embed** per modality:
  - RGB: Conv2d(3, 192, kernel=16, stride=16) → 14×14 patches for 224×224 input
  - IR: Conv2d(1, 192, kernel=16, stride=16)
  - Audio-spec: Conv2d(1, 192, kernel=16, stride=16) on mel-spectrogram
- **Modality-type tokens**: learnable (1, 1, 192) added to each modality's patches
  so the transformer knows the source
- **CLS token** prepended for classification
- **Transformer encoder**: 6 layers × 6 heads, FFN ratio 4×, GELU, dropout 0.1
- **Heads**: linear classifier + scalar confidence (sigmoid)

### Training recipe

- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Schedule: cosine annealing over 10 epochs
- Data augmentation: random crop/flip on RGB and IR (synchronized), SpecAugment on audio
- **Modality dropout during training**: independent per-modality 20% null probability
  with guaranteed ≥1 modality present

### Why missing-modality robustness matters

At inference time, sensors fail: IR camera occluded, microphone broken, RGB blinded.
Training with modality dropout forces the model to learn redundant representations
across modalities. The **meta-learned modality gate** (see fusion section) dynamically
upweights reliable sensors.

---

## Stage 3 in detail

### Input
For each detected candidate drone:
- 128-d Stage 2 embedding (from the CLS token)
- 3-d position (meters, world frame)
- 3-d velocity (m/s)
- 3-d per-modality confidence vector

→ 137-d per-node feature

### Adjacency construction

`A[i,j] = (1-α)·exp(-||pos_i - pos_j|| / σ) + α·cos(v_i, v_j)`

where α ∈ [0,1] trades spatial proximity vs. velocity coherence. Diagonal zeroed.

### Message passing

Three custom GraphLayer layers (custom implementation so the package runs without
torch-geometric). Each layer: pair-wise message MLP + sum aggregation + residual
LayerNorm.

### Outputs

- `threat_head` → sigmoid × 5 → threat score ∈ [0, 5]
- `intent_head` → softmax over 5 classes: benign / surveillance / smuggling / attack / unknown
- `action_head` → softmax over 5 classes: monitor / alert / track / jam / intercept

---

## Asynchronous fusion

### The problem

Real sensors produce data at wildly different rates:

| Sensor | Typical rate |
|---|---|
| RGB camera | 30-60 fps |
| IR camera | 30-60 fps |
| Audio | 16 kHz (raw); ~30 Hz (feature frames) |
| RF SDR | 10-100 MHz baseband |
| Radar | 10-30 Hz |
| LiDAR | 10 Hz |

Forcing all into a single "unified tensor" (as arXiv 2511.15312 does with
shape `(800, 1000, 128)`) means either downsampling the fast sensors or
interpolating the slow ones — both of which lose information.

### Our solution: AsyncSensorBuffer

- One ring buffer per modality, sized to 2 s × native rate
- Each `SensorFrame` carries a monotonic timestamp
- `snapshot(at_timestamp_ns)` returns the closest-in-time frame from each modality
- `sensor_health()` reports per-sensor rate, staleness, and mean SNR

The cascade reads a snapshot at its *own* rate (typically tied to the slowest
critical sensor, e.g., radar at 10 Hz) without forcing the fast sensors to
downsample.

---

## Meta-learned modality gate

### Context vector (14-d)

- 4-d per-modality SNR (rgb, ir, audio, rf)
- 4-d per-modality recency (1 / (1 + age_ms/100))
- 2-d hour-of-day sin/cos
- 1-d ambient light proxy
- 1-d weather proxy (inverse of RF noise floor + IR gradient)

### Architecture

```
context (14) → Linear(14, 32) → GELU → Linear(32, 32) → GELU → Linear(32, 4) → softmax
```

### How the gate is used

The 4-d softmax multiplies per-modality attention in Stage 2 (scaling the value
projections). During training, the gate is co-trained with the rest of the model
using the task loss. No auxiliary loss needed; the gate learns that "when ambient
light < 0.2, upweight IR and audio, downweight RGB" purely from task performance.

### Comparison to naive modality dropout

| Approach | How it picks modalities |
|---|---|
| Random dropout (Dai et al. 2024) | Coin flip at train time; fixed at inference |
| Fixed weights | Manually tuned per deployment |
| **Meta-learned gate (ours)** | Learned context-to-weights function |

---

## Per-modality explainability

### Stage 1 exits
Use `shap.TreeExplainer` on the XGBoost model. Group features by modality
(rf/audio) and sum absolute SHAP contributions. Report:

- Top-5 features with signed contribution
- Aggregate "RF trust" and "audio trust" percentages

Example output:
```
Stage 1 decision (conf=0.97). Drone detected.
Primary driver: Power in 2.4 GHz band (RF, toward drone, 43% of weight).
Secondary: Propeller band power (AUDIO, toward drone, 28%).
Tertiary: MFCC proxy band 3 (AUDIO, toward drone, 11%).
```

### Stage 2 exits
Use attention rollout (Abnar & Zuidema 2020) across transformer layers, then
aggregate by modality-token source. Report which modality's tokens contributed
most to the CLS representation.

### Stage 3 exits
Per-node SHAP via `shap.DeepExplainer` on the GNN, plus edge-importance via
input-gradient attribution on the adjacency matrix. Highlights which neighbors
drove a per-drone threat decision.

---

## Deployment profiles

### Edge-only (Raspberry Pi 4, ~$50)
- **Stage 1 only**
- Measured: p50 ~5 ms on Pi 4
- Accuracy ceiling: Stage 1's ~95% on DroneRF
- Use case: perimeter monitoring at rural installations

### Edge + cloud (Jetson Nano + cloud backend)
- Stage 1 on Jetson
- Uncertain cases escalate via 4G/5G to cloud-hosted Stages 2 and 3
- Network adds 20-80 ms to escalated samples
- Use case: mobile deployments, border monitoring

### On-premises (single GPU workstation)
- All three stages local
- p50: ~3 ms (Stage 1 exits), p99: ~80 ms (Stage 3 invoked)
- Use case: airports, stadiums, critical infrastructure
