# HYDRA-Net: Research Paper Outline

Target venues (in preference order):
1. **IEEE Access** — open access, fast review (~3 months), good for applied ML systems
2. **MDPI Sensors** — strong fit for multimodal sensor fusion
3. **IEEE IoT Journal** — if framed around IoT/edge deployment
4. **IROS / ICRA** — if robotics/real-time framing is emphasized

---

## Working Title

**HYDRA-Net: A Cascaded Asynchronous-Fusion Architecture for Real-Time Counter-UAV Detection with Operator-Ready Threat Assessment**

---

## Abstract (target: 200-250 words)

Counter-unmanned aerial vehicle (C-UAV) systems increasingly rely on multimodal
sensor fusion combining RF, acoustic, RGB, and thermal-IR sensors. Existing
state-of-the-art approaches treat all inputs uniformly: a single heavyweight
transformer or fusion network processes every input, regardless of how trivial
or ambiguous it is. This design inflates median latency and wastes compute on
the 90% of inputs (empty sky, obvious birds, unambiguous drone detections) that
could be resolved by much simpler models.

We introduce **HYDRA-Net**, a three-stage cascaded architecture for C-UAV
detection that uses confidence-gated early exit to route each input through the
minimum necessary compute. Stage 1 is an XGBoost classifier over handcrafted
RF and audio features (~2 ms). Stage 2 is a cross-modal attention transformer
over RGB, IR, and audio-spectrogram inputs, invoked only for Stage-1-uncertain
cases. Stage 3 is a graph neural network for swarm reasoning, invoked for
multi-target scenes. Four additional contributions: (i) asynchronous multi-rate
sensor fusion with native-rate buffering, (ii) meta-learned modality gating that
outperforms random modality dropout, (iii) operator-ready outputs including
threat score, intent class, and recommended action, and (iv) per-modality
SHAP-based decision traces for regulatory compliance.

On [dataset], HYDRA-Net achieves median latency of [X] ms (vs. [Y] ms for a
monolithic multimodal transformer baseline) while matching or exceeding
classification accuracy. The cascade's computational savings scale with deployment
context: empty-sky rural monitoring exits almost entirely at Stage 1, while urban
airport settings invoke deeper stages more often.

---

## 1. Introduction

### 1.1 Context
- Counter-UAV market growth ($4.9B → $36.4B projected by 2035)
- Use cases: airports, stadiums, critical infrastructure, military perimeter
- Sensor diversity: RF, acoustic, RGB, IR, radar, LiDAR

### 1.2 Problem with current SOTA
- Monolithic multimodal transformers (cite: Nov 2025 radar+RGB+IR+audio Transformer)
- UAUTrack, MambaSTS, DGE-YOLO — all assume full-rate synchronous fusion
- Two silent assumptions that break in production:
  1. All inputs deserve the same compute budget
  2. All sensors produce perfectly time-aligned data

### 1.3 Contributions
1. Cascaded early-exit architecture with three confidence-gated stages
2. Asynchronous multi-rate fusion
3. Meta-learned modality gating (vs. random modality dropout)
4. Threat + intent + action output (vs. detection-only)
5. Per-modality SHAP-style explainability

---

## 2. Related Work

### 2.1 Multimodal C-UAV detection
- **Audio-visual fusion**: Busset et al. 2015; HAL 2024 "Sound to Sight" (YOLOv5 + CRNN + MFCC)
- **RF fingerprinting**: Al-Sa'd et al. 2019 (DroneRF); YAMNet-based Mel-spectrogram RF (2023)
- **Thermal + visible**: MDPI 2025 "Visible-infrared multi-modal fusion" with EADW + Dempster-Shafer
- **Transformer fusion**: arXiv 2511.15312 (Nov 2025) multimodal Transformer on audio+IR+RGB+radar
- **UAUTrack / MambaSTS** (Dec 2025): vision-language tracking

### 2.2 Cascade architectures
- BranchyNet, MSDNet (generic early-exit)
- None for C-UAV, to our knowledge

### 2.3 Missing modalities and robustness
- Modality dropout (Dai et al. 2024)
- Known trade-off: excessive dropout → modality bias

### 2.4 Gap analysis (what nobody has done together)
- Cascade + async fusion + learned modality gating + operator-ready output + explainability

---

## 3. Architecture

### 3.1 Cascade overview
[Figure 1: HYDRA-Net architecture diagram]
- Stage 1: XGBoost over 30-d handcrafted features
- Stage 2: Cross-modal transformer (192-d, 6 heads, 6 layers)
- Stage 3: Message-passing GNN

### 3.2 Stage 1: fast triage
- Feature engineering (Section 3.2.1)
- Confidence calibration
- Exit criterion: confidence ≥ τ₁

### 3.3 Stage 2: cross-modal transformer
- Per-modality patch encoders
- Modality-type tokens
- Shared transformer encoder
- Dual-head output (class + calibrated confidence)

### 3.4 Stage 3: swarm GNN
- Node features: Stage 2 embedding + kinematics + per-modality confidence
- Edge construction from spatial proximity + velocity coherence
- Per-drone threat/intent/action heads

### 3.5 Asynchronous fusion layer
- Ring buffers per sensor
- Time-aligned snapshot queries
- Health monitoring

### 3.6 Meta-learned modality gate
- Context features: per-modality SNR, staleness, time-of-day, ambient light, weather proxy
- Output: softmax weights over modalities
- Training: gate co-trained with Stage 2/3

### 3.7 Explainability
- Stage 1: SHAP over feature vector, grouped by modality
- Stage 2: attention rollout
- Stage 3: per-node SHAP + edge importance

---

## 4. Experiments

### 4.1 Datasets
- Stage 1: DroneRF (Al-Sa'd et al.)
- Stage 2: Anti-UAV (CVPR 2023 Challenge)
- Stage 3: Synthetic swarm scenes + augmentations

### 4.2 Baselines
- Monolithic multimodal transformer (our re-implementation of arXiv 2511.15312)
- Single-modality baselines (XGBoost alone, RGB-only YOLO, audio CRNN)
- Naive modality dropout

### 4.3 Metrics
- Classification: accuracy, F1, ROC-AUC, per-class confusion
- Latency: p50, p90, p99, mean, std
- Cascade-specific: exit rate per stage, accuracy-among-exits
- Robustness: accuracy under simulated modality failures
- Operator metrics: threat-score calibration, intent-prediction precision

### 4.4 Main results
[Table 1: accuracy and latency across methods]
[Figure 2: latency distributions cascade vs monolithic]
[Figure 3: exit rate per stage across scenario types]

### 4.5 Ablations
- Without Stage 1 (always start at Stage 2)
- Without Stage 3 (single-drone only)
- Random modality dropout vs. learned gate
- Different confidence thresholds τ₁, τ₂

### 4.6 Explainability case studies
[Figure 4: example decisions with SHAP traces]

---

## 5. Deployment considerations

### 5.1 Edge deployment
- Stage 1 alone runs on Raspberry Pi 4 (~5 ms/sample, 10 MB RAM)
- Stage 2 requires Jetson Nano or better (~30 ms/sample at INT8)
- Stage 3 can run on same device as Stage 2

### 5.2 Failure modes
- Single-sensor failure: handled by gate
- Multiple-sensor failure: cascade falls back to remaining stages
- Adversarial inputs: addressed by Stage 3 when Stage 2 uncertainty is high

### 5.3 Regulatory and ethical considerations
- Explainability for aviation/airport deployment
- Privacy: no facial recognition in any stage
- Dual-use note

---

## 6. Conclusion and future work

- Cascade + async fusion + gating + explainability = novel contribution set
- Future: real-swarm dataset collection, RL-based intercept planning, online
  adaptation of confidence thresholds based on operator feedback

---

## Reproducibility

Code and pretrained models available at `github.com/khaled-metwalie/hydra-net` (MIT License).
Colab notebooks reproduce all figures and tables.

---

## Suggested reviewer answers (prepare in advance)

**Q: Why XGBoost for Stage 1, not a tiny neural net?**
A: XGBoost is faster, more accurate on tabular/handcrafted features, and — critically —
   its feature importance is directly interpretable for SHAP. Empirical comparison
   in Appendix A.

**Q: Isn't the cascade just ensembling?**
A: No. Ensembling runs all models and combines outputs. The cascade conditionally
   runs the next stage only if needed, which is the source of the latency advantage.

**Q: What about adversarial drones that spoof Stage 1 features?**
A: Real threat. Addressed by Stage 3's GNN, which looks at relational features
   adversaries can't easily control. Also future work on adversarial training.

**Q: How do you set τ₁ and τ₂?**
A: Calibrated on held-out validation data by Pareto-optimizing (latency, accuracy).
   We report results across the Pareto frontier in Section 4.
