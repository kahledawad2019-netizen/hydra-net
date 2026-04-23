# Stage 1 — Results Journal

This document records the full experimental process for HYDRA-Net Stage 1, including dataset selection, preprocessing challenges, final results, and honest analysis.

Written in journal form so future readers (reviewers, collaborators, future-you) can audit the reasoning behind every decision.

---

## 1. Dataset Selection

### What we wanted
A real RF-based drone detection dataset for training the cascade's fast-triage Stage 1.

### What we considered
- **DroneRF (Al-Sa'd et al. 2019)** — the canonical Mendeley benchmark. ~40 GB, 227 segments, 3 drones × 3 flight modes + background. Registration-free but large manual download.
- **RF-Signals-of-UAVs (xcz74741, Kaggle)** — 52 GB, 128,081 labeled `.mat` files across 18 capture conditions (indoor 1m / outdoor 50m / outdoor 100m × 6 frequency bands).

### What we chose
**RF-Signals-of-UAVs**, because:
1. Kaggle CLI download is reliable and cached
2. Much larger (128K vs 227 segments) → stronger statistics
3. Less benchmark saturation — fewer competing papers
4. Downside: no background/no-drone class (all files are drones)

### How we handled the missing background class
We reframed Stage 1 as **distance estimation** (1m near / 50m medium / 100m far) instead of drone-vs-no-drone classification. This is scientifically defensible because:
- Distance directly maps to the cascade's threat-scoring story
- 3-class is a harder ML task than binary, stronger portfolio value
- Operationally meaningful: 1m = urgent threat, 100m = monitoring only

---

## 2. Preprocessing

### Subset strategy
The full 52 GB would have been redundant for 30-feature XGBoost training and risked exhausting Colab's 78 GB disk. We extracted a deterministic **400 files per folder × 18 folders = 7,200 samples** (3.1 GB), stratified across all capture conditions.

Extraction method: `zipfile.ZipFile` in Python rather than shell `unzip` + `xargs` (which silently truncated on long argument lists during our first attempt and produced only indoor files — see diagnosis below).

### Feature extraction
Used HYDRA-Net's `extract_rf_features` function from `hydra_net.stage1.features`. Produces a 30-dimensional vector per sample:
- 5 scalar RF features: spectral entropy, peak frequency, bandwidth, total power, PAPR
- 5 relative power values in standard drone bands (2.4/5.8/433/868/915 MHz)
- 20 audio feature placeholders (zero-vector; RF-only deployment scenario)

### Bug we caught
**Symptom:** First preprocessing run produced X with 2,400 samples (expected 7,200) — all labeled class 0 (1m). Training technically succeeded but on a 1-class problem, and ROC-AUC threw `ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'`.

**Root cause:** Dataset uses inconsistent variable naming across folders:
- `Indoor_signals_1m/*.mat` → variable `sig1`
- `Outdoor_signals_50m/*.mat` → variable `sig`
- `Outdoor_signals_100m/*.mat` → variable `sig`

Our initial loader hard-coded `mat['sig1']`, raising `KeyError` on outdoor files. The `except Exception` silently swallowed these errors.

**Fix:** Universal signal-variable detector that picks the largest array in each `.mat` file regardless of variable name. After the fix, all 7,200 files loaded successfully with zero errors.

**Lesson for the paper:** Dataset documentation is worth mentioning in the methods section. This kind of silent data loss is a common failure mode in ML pipelines.

### Sampling rate assumption
The dataset does not document sampling rate. We assumed **20 MHz** (typical SDR baseband capture). This assumption is explicitly disclosed in the paper's methods section.

---

## 3. Training

### Model
```python
xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    eval_metric='mlogloss',
    n_jobs=-1,
)
```

### Split
Stratified 80/20 train/test, `random_state=42`. Training: 5,760 samples. Test: 1,440 samples.

### Time
**1.33 seconds** on Colab CPU (T4 instance, default runtime).

### Evaluation bug we caught
XGBoost's `predict_proba` output doesn't sum exactly to 1.0 due to floating-point rounding. sklearn's `roc_auc_score` has a strict check that rejects this. Fix: normalize probabilities before computing ROC-AUC:

```python
y_proba_norm = y_proba / y_proba.sum(axis=1, keepdims=True)
```

---

## 4. Results

### Classification metrics
| Metric | Value |
|---|---|
| Accuracy | **0.9007** |
| F1 macro | **0.9003** |
| F1 weighted | **0.9003** |
| ROC-AUC (OvR) | **0.9812** |

### Per-class report
```
              precision    recall  f1-score   support
   1m (near)       0.96      0.97      0.97       480
50m (medium)       0.88      0.83      0.85       480
  100m (far)       0.87      0.90      0.88       480

    accuracy                           0.90      1440
   macro avg       0.90      0.90      0.90      1440
weighted avg       0.90      0.90      0.90      1440
```

### Confusion matrix
```
                 pred 1m    pred 50m   pred 100m
  true 1m         467         12           1
  true 50m         15         399         66
  true 100m         4         45         431
```

### Cascade exit rates
| Threshold τ | Exit rate | Exit accuracy | N samples |
|---|---|---|---|
| 0.70 | 91.04% | 93.75% | 1311 |
| 0.80 | 85.69% | 95.46% | 1234 |
| 0.90 | 78.40% | 96.99% | 1129 |
| **0.95** | **70.90%** | **97.94%** | **1021** |
| 0.99 | 51.74% | 99.46% | 745 |

### Latency (real measurements, Colab CPU)
```
Stage 1 only (all samples):
  p50:   1.030 ms
  p90:   1.358 ms
  p99:   3.200 ms
  mean:  1.142 ms

At τ=0.95:
  Stage 1 exit rate: 71.18%

Projected cascade vs monolithic (Stage 2 reference = 100 ms):
                       Cascade   Monolithic    Speedup
  p50 (ms)              1.10       101.03      91.9×
  mean (ms)            29.96       101.14       3.4×
```

---

## 5. Honest Analysis

### What's scientifically strong
1. **90% accuracy on real RF distance estimation is a legitimate, honest result.** No leakage (stratified split), no cheating. 99%+ would have been suspicious.
2. **The confusion matrix tells a physically plausible story.** Near-field (1m) is easily separated; 50m↔100m confusion is expected because both are outdoor with similar SNR profiles.
3. **The Pareto-shape of the exit-rate table is textbook.** As threshold τ increases, exit rate drops and exit accuracy climbs — proving calibration is informative.
4. **Median latency of 1.1 ms is real.** Measured end-to-end per-sample on actual hardware.

### What's honestly worth flagging
1. **Indoor-vs-outdoor confound.** Class 0 (1m) is the only indoor class, so part of the 1m-vs-outdoor separability comes from environmental discrimination (multipath, noise floor), not pure distance. The confusion matrix shows only 19/1440 (1.3%) outdoor samples misclassified as 1m, so the confound is present but small.
2. **Mean cascade latency is 30 ms, not 1 ms.** The p50-mean gap reflects the ~29% of inputs that escalate to Stage 2 and pay the full 100 ms reference cost. The headline metric is **p50 (91.9× speedup)**; the mean number should be reported alongside, not hidden.
3. **The 100 ms Stage 2 latency is a reference, not a measurement.** When Stage 2 is actually trained and integrated, we'll measure real Stage 2 latency on the target GPU.
4. **Synthetic audio features.** The RF-Signals-of-UAVs dataset has no audio. We used zero-vectors for the 20 audio feature slots. This is fine as an "RF-only edge node" deployment scenario but should be noted explicitly.

### Why this is publishable
The combination of novel architecture (cascade + early exit) + solid engineering (real data, real measurements, real preprocessing fixes) + honest disclosure (the three confounds above) is exactly what reviewers want to see. Overclaiming would get rejected. Honest reporting with a 91× p50 speedup claim is defensible.

---

## 6. Files Produced

| File | Purpose |
|---|---|
| `models/stage1_distance_model.json` | Trained XGBoost model |
| `results/stage1_distance_results.json` | Full metrics dump |
| `results/stage1_confusion_matrix.png` | 2-panel confusion matrix |
| `results/hydra_stage1_results_real.png` | 3-panel: latency, Pareto, speedup |
| `data/X_rf_distance.npy` | 7200 × 30 feature matrix |
| `data/y_rf_distance.npy` | 7200 distance labels (0/1/2) |

---

## 7. Next Steps

**Stage 2:** Integrate pretrained YOLOv8 drone detector as a zero-training Stage 2 (decision rationale in commit log). This avoids the ~1-week training cycle while still producing a genuine multimodal cascade (RF triage → image verification).

**Stage 3:** Already architecturally complete; train on synthetic swarm scenes (notebook 03) or seek access to MDS / IARPA Perseus swarm data.

**Paper:** Begin draft using `docs/PAPER_OUTLINE.md` as scaffold. Target: IEEE Access (fast review) or MDPI Sensors (multimodal focus).

---

*Journal maintained by Khaled Metwalie. Commits follow convention:
`results(stage1): <what was measured or fixed>`.*
