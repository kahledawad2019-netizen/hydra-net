# HYDRA-Net

**H**ierarchical **Y**ield-**D**riven **R**esilient **A**sync-fusion Network for Counter-UAV Detection

> A cascaded multimodal architecture for anti-drone systems that achieves low median latency through confidence-gated early exit, while retaining full multimodal reasoning capacity for hard cases (swarms, adversarial drones, ambiguous signals).

---

## ⚠️ Project Status: Research Scaffold

This repository is a **research scaffold** — the architecture, training pipelines, evaluation harness, and Colab-ready notebooks are complete and runnable, but the models have **not yet been trained on full real-world datasets**. Training requires GPU access (Colab Pro / Kaggle / local CUDA) and downloaded benchmark datasets.

Anyone cloning this repo can:
1. Run the synthetic-data proof-of-concept locally on CPU (demonstrates the cascade mechanism)
2. Open the provided Colab notebooks to train each stage on real datasets (DroneRF, MPED-RF, Anti-UAV)
3. Reproduce the latency benchmarks (these are real — measured on actual code)
4. Extend the architecture with new sensors or cascade stages

Results reported in `results/` are **synthetic-PoC results** clearly labeled as such. Real-world benchmark results will be added as training completes.

---

## 🎯 Core Idea in One Paragraph

Existing multimodal counter-UAV systems (e.g., Transformer fusion of radar+RGB+IR+audio, published 2025) run their full expensive model on every input — whether it's a bird, an empty sky, or a coordinated swarm attack. HYDRA-Net is a **three-stage cascade**: a fast XGBoost triage handles the 90% of trivial inputs in ~2 ms, a small cross-modal transformer handles uncertain single-drone cases in ~15 ms, and a deep fusion + GNN stage handles swarms and adversarial cases in ~60 ms. Median latency drops from ~200 ms to ~3 ms; worst-case latency remains competitive with SOTA. The system outputs not just detection, but threat level, predicted intent, and explainability attributions per modality.

## 🔑 Five Novel Contributions

1. **Confidence-gated cascade with early exit** — not published for counter-UAV to date
2. **Asynchronous multi-rate fusion** — handles sensors at native rates (audio 16 kHz, radar 10 Hz, etc.) without forced synchronization
3. **Meta-learned modality gating** — learns *which sensor to trust under which conditions* (trust audio in fog, distrust RGB at night), rather than fixed weights or random dropout
4. **Threat + intent output head** — operator-ready outputs (threat 0-5, predicted trajectory, recommended action), not just bounding boxes
5. **Per-modality SHAP-style explainability** — auditable decision trace, critical for regulatory acceptance in airports/critical infrastructure

## 📊 Competitive Positioning

| Feature | SOTA Multimodal Transformer (Nov 2025) | HYDRA-Net |
|---|---|---|
| Sensor fusion | Synchronous | **Asynchronous, multi-rate** |
| Median latency (target) | 100-300 ms | **~3 ms** |
| P99 latency (target) | 100-300 ms | ~80 ms |
| Missing sensor handling | Naive dropout | **Meta-learned gating** |
| Swarm reasoning | Partial | **GNN-based Stage 3** |
| Output | Bounding box + class | **Threat + intent + action** |
| Explainability | Black-box | **SHAP per modality** |
| Edge deployable (Raspberry Pi) | No | **Yes — Stage 1 only** |

Note: latencies for HYDRA-Net are design targets validated on CPU with synthetic features. Real-dataset validation is pending GPU training.

## 📁 Repository Structure

```
hydra-net/
├── src/hydra_net/
│   ├── stage1/          # XGBoost fast triage
│   ├── stage2/          # Cross-modal transformer
│   ├── stage3/          # Deep fusion + GNN for swarms
│   ├── fusion/          # Async multi-rate fusion, modality gating
│   └── explainability/  # SHAP-style per-modality attribution
├── notebooks/
│   ├── 01_stage1_dronerf_colab.ipynb    # Train Stage 1 on DroneRF
│   ├── 02_stage2_antiuav_colab.ipynb    # Train Stage 2 on Anti-UAV
│   ├── 03_stage3_swarm_colab.ipynb      # Train Stage 3 on swarm data
│   └── 04_end_to_end_benchmark.ipynb    # Full cascade evaluation
├── configs/             # YAML configs per stage
├── data/
│   ├── raw/             # Download location for real datasets
│   ├── processed/       # Preprocessed features
│   └── synthetic/       # Synthetic PoC data
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_stage1.py
│   ├── benchmark_latency.py
│   └── download_datasets.sh
├── tests/               # Unit tests
├── results/             # Logged experiment results
├── models/              # Saved model checkpoints
└── docs/
    ├── ARCHITECTURE.md  # Detailed design
    ├── DATASETS.md      # Dataset acquisition guide
    └── PAPER_OUTLINE.md # Research paper scaffold
```

## 🚀 Quickstart

### Local PoC (CPU, no GPU needed)
```bash
git clone <this-repo>
cd hydra-net
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python scripts/train_stage1.py --synthetic
python scripts/benchmark_latency.py
```

### Train on Real Data (GPU recommended)
Upload `notebooks/01_stage1_dronerf_colab.ipynb` to Google Colab or Kaggle, follow the in-notebook dataset download instructions, and run all cells. Repeat for Stage 2 and Stage 3 notebooks.

## 📚 Datasets (not included — see `docs/DATASETS.md`)

| Dataset | Modality | Size | Purpose |
|---|---|---|---|
| DroneRF | RF | ~40 GB | Stage 1 RF features |
| MPED-RF | RF | ~15 GB | Stage 1 augmentation |
| DroneAudioDataset | Audio | ~2 GB | Stage 1 acoustic features |
| Anti-UAV (CVPR challenge) | RGB+IR | ~80 GB | Stage 2 vision |
| UAV-Eagle | RGB | ~5 GB | Stage 2 augmentation |
| MDS (multi-drone swarm, synthetic) | all | ~30 GB | Stage 3 swarm |

## 🔬 Research Paper Status

Paper outline is in `docs/PAPER_OUTLINE.md`. Target venues:
- IEEE Access (open access, fast review)
- Sensors (MDPI, multimodal focus)
- IROS / ICRA (robotics conferences, if real-time robotics framing)

## 👤 Author

**Khaled Metwalie** — Data Scientist, DEPI graduate
Portfolio: [khaledmetwalie.lovable.app](https://khaledmetwalie.lovable.app)

## 📄 License

MIT — see LICENSE file.

## 🤝 Acknowledgments

Built with architectural guidance on the cascade early-exit idea and competitive analysis of counter-UAV SOTA as of April 2026. All implementation, training, and results generated by the author.
