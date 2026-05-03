# HYDRA-Net

**H**ierarchical **Y**ield-**D**riven **R**esilient **A**sync-fusion Network for Counter-UAV Detection

> A cascaded multimodal architecture for anti-drone systems that achieves low median latency through confidence-gated early exit, while retaining full multimodal reasoning capacity for hard cases.

---

## 🎯 Measured Results (Real Data)

**Stage 1 trained and validated on the RF-Signals-of-UAVs dataset** (xcz74741, Kaggle, 128K labeled RF segments from 6 frequency bands × 3 distances × indoor/outdoor).

### Classification performance
| Metric | Value |
|---|---|
| Task | Drone distance estimation (1m / 50m / 100m, 3-class) |
| Training samples | 5,760 |
| Test samples | 1,440 |
| **Accuracy** | **90.07%** |
| **F1 (macro)** | **0.9003** |
| **ROC-AUC (OvR)** | **0.9812** |

### Cascade behavior (the novel contribution)
| Confidence threshold τ | Exit rate | Accuracy among exits |
|---|---|---|
| 0.70 | 91.04% | 93.75% |
| 0.80 | 85.69% | 95.46% |
| 0.90 | 78.40% | 96.99% |
| **0.95 (operating point)** | **70.90%** | **97.94%** |
| 0.99 | 51.74% | 99.46% |

### Latency (real measurements on Colab CPU)
| | HYDRA-Net cascade | Monolithic baseline | Speedup |
|---|---|---|---|
| **p50** | **1.10 ms** | 101.03 ms | **91.9×** |
| p90 | 1.36 ms | 101.36 ms | — |
| Mean | 29.96 ms | 101.14 ms | 3.4× |

The p50-vs-mean gap reflects the cascade's asymmetric cost structure: 71% of inputs exit at Stage 1 in ~1 ms, while the 29% that escalate to Stage 2 pay the full multimodal cost (~100 ms reference). This is the intended behavior.

### Confusion matrix

```
                 pred 1m    pred 50m   pred 100m
  true 1m         467         12           1      (97% recall)
  true 50m         15         399         66      (83% recall)
  true 100m         4         45         431      (90% recall)
```

Near-field (1m) signals are near-perfectly discriminated. The 50m↔100m confusion is the expected hard case — both are outdoor, similar SNR — and explicitly motivates the cascade's Stage 2 escalation design.

See `results/stage1_distance_results.json` for the full metrics dump, and `results/hydra_stage1_results_real.png` for Pareto and latency plots.

---

## ⚠️ Status & Honest Disclosure

**What's complete:**
- All architecture code (Stages 1–3, async fusion, meta-learned modality gate, explainability, FastAPI serving)
- Stage 1 trained on real RF data (results above)
- Latency benchmark on real data
- Full documentation, 4 Colab notebooks, 14 passing unit tests

**What's pending:**
- Stage 2 on the Anti-UAV RGB+IR benchmark (architecture implemented; training requires GPU + dataset registration). Plan: integrate a pretrained YOLOv8 drone detector as Stage 2 in the next iteration.
- Stage 3 (swarm-reasoning GNN) trained only on synthetic swarm scenes; no public labeled swarm dataset exists.

**Data honesty notes:**
- The RF-Signals-of-UAVs dataset contains only drone signals (no background / no-drone class). We reframed Stage 1 as distance estimation rather than drone-vs-no-drone classification.
- Class 0 (1m) is the only indoor class. Part of the 1m-vs-outdoor separability comes from indoor/outdoor environmental discrimination, not pure distance. Confusion matrix shows only 19/1440 (1.3%) outdoor samples misclassified as 1m, so this confound is present but small.
- Sampling rate is assumed 20 MHz (typical SDR baseband) as the dataset metadata does not specify.
- The dataset variable naming is inconsistent across folders (`sig1` in indoor, `sig` in outdoor). Our preprocessing uses a universal signal-variable detector.

---

## 🔑 Five Novel Contributions

1. **Confidence-gated cascade with early exit** — not published for counter-UAV to date
2. **Asynchronous multi-rate fusion** — sensors at native rates (audio 16 kHz, radar 10 Hz, etc.)
3. **Meta-learned modality gating** — context-aware trust weights, not random dropout
4. **Threat + intent output head** — operator-ready (threat 0–5, trajectory, recommended action)
5. **Per-modality SHAP-style explainability** — auditable decision trace for regulatory acceptance

---

## 📊 Competitive Positioning

| Feature | SOTA Multimodal Transformer (Nov 2025) | HYDRA-Net |
|---|---|---|
| Sensor fusion | Synchronous | **Asynchronous, multi-rate** |
| Median latency (measured) | ~100 ms (reference) | **1.10 ms** |
| Missing sensor handling | Naive dropout | **Meta-learned gating** |
| Swarm reasoning | Partial | **GNN-based Stage 3** |
| Output | Bounding box + class | **Threat + intent + action** |
| Explainability | Black-box | **SHAP per modality** |
| Edge deployable (Raspberry Pi) | No | **Yes — Stage 1 only** |

---

## 📁 Repository Structure

```
hydra-net/
├── src/hydra_net/
│   ├── cascade.py              orchestrator · confidence-gated early exit
│   ├── stage1/                 XGBoost fast triage (30-d handcrafted features)
│   ├── stage2/                 cross-modal transformer
│   ├── stage3/                 swarm-reasoning GNN
│   ├── fusion/                 async buffers + meta-learned gate
│   ├── explainability/         per-modality SHAP traces
│   └── serving/                FastAPI endpoints
├── notebooks/                  4 Colab notebooks for real-dataset training
├── scripts/                    synthetic data, training, benchmarking
├── tests/                      14 passing unit tests
├── docs/                       ARCHITECTURE.md · DATASETS.md · PAPER_OUTLINE.md
├── results/                    real-data results + plots
└── configs/                    YAML hyperparameters
```

---

## 🚀 Quickstart

### Local PoC (CPU, no GPU needed)
```bash
git clone https://github.com/YOUR-USERNAME/hydra-net.git
cd hydra-net
pip install -r requirements.txt
PYTHONPATH=src python scripts/quickstart_demo.py
```

### Reproduce Stage 1 results on Colab
Open `notebooks/01_stage1_dronerf_colab.ipynb` in Google Colab. Follow the dataset download instructions (Kaggle RF-Signals-of-UAVs `xcz74741/rf-signals-of-uavs`) and run all cells.

### Serve via API
```bash
uvicorn hydra_net.serving.api:app --port 8000
```

---

## 📚 Datasets

| Dataset | Stage | Size | Access |
|---|---|---|---|
| **RF-Signals-of-UAVs** (used) | 1 | 52 GB (3 GB subset used) | Kaggle: `xcz74741/rf-signals-of-uavs` |
| Anti-UAV (CVPR Challenge) | 2 | ~80 GB | Registration at anti-uav.github.io |
| Synthetic swarm scenes | 3 | procedural | generator in notebook 03 |

Details: `docs/DATASETS.md`.

---

## 👤 Author

**Khaled Metwalie**&**rahma ahmed** — Data Scientist, DEPI graduate
Portfolio: [khaledmetwalie.lovable.app](https://khaledmetwalie.lovable.app)

## 📄 License

MIT — see `LICENSE`.

## 📖 Citation

If you use this architecture in research, cite (preprint in preparation):

```bibtex
@misc{metwalie2026hydranet,
  author = {Khaled Metwalie},
  title  = {HYDRA-Net: A Cascaded Asynchronous-Fusion Architecture for Real-Time Counter-UAV Detection},
  year   = {2026},
  url    = {https://github.com/YOUR-USERNAME/hydra-net}
}
```
