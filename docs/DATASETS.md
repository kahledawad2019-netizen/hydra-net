# Datasets Guide

This document describes the datasets used by HYDRA-Net, along with download
instructions, preprocessing notes, and licensing constraints.

All datasets are used for research purposes only. Respect each dataset's license.

---

## Stage 1 — RF and Audio Datasets

### DroneRF (primary Stage 1 dataset)
- **Paper:** Al-Sa'd et al., "RF-based drone detection and identification using deep learning approaches," *Future Generation Computer Systems*, 2019.
- **Data:** Raw RF captures (both control and video channels) from 3 consumer drones (DJI Phantom 4 Pro, Parrot Bebop 1, Parrot AR Drone) across 4 flight modes, plus background (no drone).
- **Size:** ~40 GB uncompressed
- **Sample rate:** 40 MHz (RTL-SDR pair, 2.4 GHz band)
- **Hosting:** [Mendeley Data](https://data.mendeley.com/datasets/f4c2b4n755/1)
- **License:** CC BY 4.0
- **Download:**
  ```bash
  # Visit the Mendeley Data link above. A direct-download JSON API is
  # available; see their documentation for a stable link.
  # Once downloaded, the structure is:
  # dronerf/
  #   00000L_0.csv, 00000L_1.csv, ...        # background (no drone), low band
  #   00000H_0.csv, ...                       # background, high band
  #   10000L_*.csv, 10000H_*.csv             # Bebop, mode 0
  #   10001L_*.csv, 10001H_*.csv             # Bebop, mode 1
  #   ...                                     # see dataset readme for full code
  ```
- **Classes in DroneRF code format:**
  - `00000`: no drone (background)
  - `1xxxx`: Bebop
  - `10x0x`: AR Drone (not present in original release)
  - `11xxx`: Phantom
  - Last two digits of the 5-digit code encode flight mode.

### MPED-RF (augmentation)
- **Paper:** Swinney & Woods, "Multi-sensor drone detection," various.
- **Data:** Multi-channel RF recordings.
- **Size:** ~15 GB
- **Access:** Research-access via author request.

### DroneAudioDataset
- **Paper:** Al-Emadi et al., "Audio-based drone detection and identification using deep learning," 2019.
- **Data:** ~1,300 audio clips (drone vs background)
- **Size:** ~2 GB
- **Hosting:** [GitHub](https://github.com/saraalemadi/DroneAudioDataset)
- **License:** Check repo for specifics.

---

## Stage 2 — Vision Datasets

### Anti-UAV (CVPR Challenge)
- **Paper:** Zhao et al., "The 2nd Anti-UAV Workshop & Challenge: Methods and Results," CVPR 2023.
- **Data:** Synchronized RGB and thermal-IR video of drones against sky, buildings, etc. 160+ sequences, multi-sequence tracking.
- **Size:** ~80 GB (video)
- **Hosting:** [Challenge site](https://anti-uav.github.io/)
- **Registration:** Required (free, academic)
- **Label format:** Per-frame `exist` (0/1) + bounding box.
- **License:** Research only; see the challenge terms.

### UAV-Eagle (augmentation)
- **Data:** RGB UAV detection benchmark.
- **Size:** ~5 GB
- **Use:** Augment Stage 2 training when Anti-UAV is insufficient for certain drone types.

---

## Stage 3 — Swarm Datasets (scarce)

This is the dataset gap HYDRA-Net openly acknowledges. Public labeled swarm datasets
are extremely limited. Three approaches:

1. **Synthetic swarm generator** (provided in `notebooks/03_stage3_swarm_colab.ipynb`):
   procedural scene generation for formations, dispersed clusters, and attacking patterns.

2. **MDS (Multi-Drone Simulation)**: unity-based simulation; release status varies.

3. **Real multi-UAV tracking datasets**: adapt from Anti-UAV multi-sequence tracking
   or VisDrone's multi-object tracking splits.

---

## Preprocessing Pipeline

### Stage 1 (RF + audio → handcrafted features)

```python
from hydra_net.stage1 import FeatureConfig, extract_rf_features, extract_audio_features

config = FeatureConfig(rf_sample_rate=40e6)  # DroneRF rate
rf_feats = extract_rf_features(rf_samples, config)
audio_feats = extract_audio_features(audio_samples, config)
X = np.concatenate([rf_feats, audio_feats])
```

Expected feature vector length: 30 (5 RF scalars + 5 band-powers + 7 audio scalars + 13 MFCC proxy bands).

### Stage 2 (RGB + IR → tensors)

Standard image preprocessing: resize to 224×224, normalize with ImageNet stats for RGB,
single-channel normalization for IR. The `AntiUAVDataset` class in
`notebooks/02_stage2_antiuav_colab.ipynb` handles this.

### Stage 3 (detection → node features)

Each node feature is: `[Stage 2 embedding (128d), position (3d), velocity (3d), per-modality confidence (3d)] = 137d total`.

Position and velocity are in a world-frame coordinate system (meters, meters/second).

---

## Storage Recommendations

For Google Colab workflow, cache preprocessed features (not raw data) to Drive:

```
/content/drive/MyDrive/hydra-net-data/
├── dronerf/
│   ├── X.npy                # preprocessed feature matrix (~50 MB)
│   └── y.npy                # labels (~1 MB)
├── antiuav/
│   └── <extracted frames>   # or keep raw MP4s and decode on-demand
├── models/
│   ├── stage1_dronerf.json
│   ├── stage2_antiuav.pt
│   └── stage3_swarm.pt
└── results/
```

Only the preprocessed features need to persist; raw data can be re-downloaded if deleted.

---

## License Summary Table

| Dataset | License | Commercial use? |
|---|---|---|
| DroneRF | CC BY 4.0 | ✅ with attribution |
| DroneAudioDataset | Check repo | Usually research-only |
| Anti-UAV | Challenge ToS | ❌ research only |
| UAV-Eagle | Varies | Check source |

For any commercial counter-UAV deployment built on HYDRA-Net, train on
licensed data (e.g., your own collected data or commercially-licensed datasets).
