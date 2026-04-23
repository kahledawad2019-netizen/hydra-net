"""
Handcrafted Feature Extraction for Stage 1
===========================================

Stage 1 trades model complexity for feature intelligence. Instead of deep
feature learning, we extract a compact, physically-motivated feature vector
from RF and audio streams.

Feature categories:
  RF:    spectral entropy, peak frequency, bandwidth, power in drone bands
  Audio: MFCC statistics, spectral centroid, propeller harmonic strength
  Meta:  ambient noise floor, SNR estimate

These features are what a domain expert would compute; the XGBoost learns
how to combine them. This is why Stage 1 can be fast *and* accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal


# Known drone RF control bands (Hz)
DRONE_RF_BANDS = [
    (2.400e9, 2.4835e9),   # 2.4 GHz ISM (most consumer drones)
    (5.725e9, 5.875e9),    # 5.8 GHz ISM (DJI, Autel)
    (433e6,   435e6),      # 433 MHz long-range
    (868e6,   870e6),      # 868 MHz (Europe)
    (915e6,   928e6),      # 915 MHz (Americas)
]

# Known propeller harmonic frequency ranges (Hz)
# Quadcopter props typically 50-300 Hz fundamental with strong harmonics
PROP_FUNDAMENTAL_RANGE = (50.0, 300.0)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    rf_sample_rate: float = 20e6       # 20 MHz SDR typical
    audio_sample_rate: int = 16000     # 16 kHz common
    n_mfcc: int = 13
    rf_band_power_bands: tuple = tuple(DRONE_RF_BANDS)


def extract_rf_features(rf_signal: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """
    Extract handcrafted features from an RF signal segment.

    Parameters
    ----------
    rf_signal : np.ndarray
        Complex or real-valued RF samples.
    config : FeatureConfig

    Returns
    -------
    np.ndarray
        1D feature vector (length depends on number of RF bands configured).
    """
    # Use magnitude for real-valued processing
    x = np.abs(rf_signal).astype(np.float32)

    # 1. Spectral entropy — drones have structured signals (low entropy vs noise)
    freqs, psd = sp_signal.welch(x, fs=config.rf_sample_rate, nperseg=min(1024, len(x)))
    psd_norm = psd / (psd.sum() + 1e-12)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # 2. Peak frequency
    peak_freq = float(freqs[np.argmax(psd)])

    # 3. Bandwidth (-3 dB width around peak)
    peak_power = psd.max()
    half_power_mask = psd >= peak_power / 2
    bandwidth = float(freqs[half_power_mask][-1] - freqs[half_power_mask][0]) if half_power_mask.any() else 0.0

    # 4. Total power
    total_power = float(psd.sum())

    # 5. Peak-to-average power ratio
    papr = float(peak_power / (psd.mean() + 1e-12))

    # 6. Power in each known drone band (relative)
    # Note: Welch's PSD gives us up to Nyquist = sample_rate / 2.
    # For real deployments with higher band analysis, signal is downconverted first.
    band_powers = []
    nyquist = config.rf_sample_rate / 2
    for low, high in config.rf_band_power_bands:
        if low > nyquist:
            # Band outside our sampling range — contributes 0 (would need downconversion)
            band_powers.append(0.0)
            continue
        mask = (freqs >= low) & (freqs <= min(high, nyquist))
        band_powers.append(float(psd[mask].sum() / (total_power + 1e-12)))

    features = np.concatenate([
        [spectral_entropy, peak_freq, bandwidth, total_power, papr],
        band_powers,
    ]).astype(np.float32)

    return features


def extract_audio_features(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """
    Extract handcrafted features from an audio segment.

    Focus on features that distinguish drone propellers from ambient and
    common false-positive sources (birds, wind, traffic).

    Parameters
    ----------
    audio : np.ndarray
        Mono audio samples.
    config : FeatureConfig

    Returns
    -------
    np.ndarray
        1D feature vector.
    """
    sr = config.audio_sample_rate
    audio = audio.astype(np.float32)

    # 1. RMS energy
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # 2. Zero-crossing rate (high for noise, low for tonal drone sounds)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    # 3. Spectral features via FFT
    n_fft = min(2048, len(audio))
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    psd = spectrum ** 2
    psd_norm = psd / (psd.sum() + 1e-12)

    # Spectral centroid — drones often have mid-frequency energy concentration
    centroid = float(np.sum(freqs * psd_norm))

    # Spectral rolloff (95%)
    cumulative = np.cumsum(psd_norm)
    rolloff_idx = np.argmax(cumulative >= 0.95)
    rolloff = float(freqs[rolloff_idx])

    # Spectral flatness (Wiener entropy) — drone propellers = tonal (low flatness)
    geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd)
    flatness = float(geo_mean / (arith_mean + 1e-12))

    # 4. Propeller harmonic strength
    # Check for periodic peaks in the propeller fundamental range
    prop_mask = (freqs >= PROP_FUNDAMENTAL_RANGE[0]) & (freqs <= PROP_FUNDAMENTAL_RANGE[1])
    prop_power = float(psd[prop_mask].sum() / (psd.sum() + 1e-12))

    # Harmonic-to-noise: check if there are distinct peaks (quadcopters
    # have 4 propellers at slightly different RPMs → multiple close peaks)
    if prop_mask.any():
        prop_psd = psd[prop_mask]
        peak_threshold = prop_psd.mean() + 2 * prop_psd.std()
        n_peaks = int(np.sum(prop_psd > peak_threshold))
    else:
        n_peaks = 0

    # 5. MFCC means (simplified — first n_mfcc mel bands)
    # For a real implementation, use librosa. This is a lightweight proxy.
    n_bands = config.n_mfcc
    band_edges = np.linspace(0, len(psd), n_bands + 1, dtype=int)
    mfcc_proxy = np.array([
        float(np.log(psd[band_edges[i]:band_edges[i + 1]].sum() + 1e-12))
        for i in range(n_bands)
    ], dtype=np.float32)

    features = np.concatenate([
        [rms, zcr, centroid, rolloff, flatness, prop_power, float(n_peaks)],
        mfcc_proxy,
    ]).astype(np.float32)

    return features


def extract_combined_features(
    rf_signal: np.ndarray,
    audio: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Concatenated RF + audio feature vector for Stage 1."""
    config = config or FeatureConfig()
    rf_feats = extract_rf_features(rf_signal, config)
    audio_feats = extract_audio_features(audio, config)
    return np.concatenate([rf_feats, audio_feats])


def feature_dim(config: FeatureConfig | None = None) -> int:
    """Return the expected feature-vector dimension for given config."""
    config = config or FeatureConfig()
    # RF: 5 scalar + n_bands band_powers
    rf_dim = 5 + len(config.rf_band_power_bands)
    # Audio: 7 scalar + n_mfcc bands
    audio_dim = 7 + config.n_mfcc
    return rf_dim + audio_dim
