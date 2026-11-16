"""
Pluggable FeatureExtractor interface and built-in implementations.

Each extractor exposes:
  name: str          — identifier used in settings
  dim: int           — output dimensionality
  extract(y, sr)     — produce a 1-D float32 numpy vector
  available() -> bool — True if all optional deps are installed
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FeatureExtractor(Protocol):
    name: str
    dim: int

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Return a fixed-length float32 feature vector for the audio segment."""
        ...

    def available(self) -> bool:
        """Return True if all optional dependencies are installed."""
        ...


# ---------------------------------------------------------------------------
# Shared windowed-stats helpers (used by MFCC and Spectral extractors)
# ---------------------------------------------------------------------------

def _stats4(arr: np.ndarray) -> list[float]:
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 90)),
    ]


def _window_coeff_stats(wins: list) -> list[float]:
    mat = np.array(wins)
    result: list[float] = []
    for col in mat.T:
        result.extend(_stats4(col))
    return result


# ---------------------------------------------------------------------------
# MFCCExtractor  (200-dim — the original baseline)
# ---------------------------------------------------------------------------

_N_MFCC = 20
_N_CONTRAST_BANDS = 4
_CONTRAST_FMIN = 100.0
_WINDOW_DURATION = 2.0
_WINDOW_HOP = 1.0


def _extract_mfcc_window(y: np.ndarray, sr: int) -> dict:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    delta_width = max(3, min(9, (mfcc.shape[1] // 2) * 2 - 1))
    if delta_width % 2 == 0:
        delta_width -= 1
    delta_width = max(3, delta_width)
    return {
        "mfcc": np.mean(mfcc, axis=1),
        "delta_mfcc": np.mean(librosa.feature.delta(mfcc, width=delta_width), axis=1),
        "contrast": np.mean(
            librosa.feature.spectral_contrast(
                y=y, sr=sr, n_bands=_N_CONTRAST_BANDS, fmin=_CONTRAST_FMIN
            ),
            axis=1,
        ),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "rms": float(np.mean(librosa.feature.rms(y=y))),
    }


class MFCCExtractor:
    """
    200-dimensional handcrafted feature vector.

    Features computed on overlapping 2-second windows then summarised with
    mean/std/10th-pct/90th-pct:
      - MFCCs (20 × 4 = 80)
      - Delta-MFCCs (20 × 4 = 80)
      - Spectral contrast (5 bands × 4 = 20)
      - Centroid, bandwidth, rolloff, ZCR, RMS (5 × 4 = 20)
    """

    name = "mfcc"
    dim = 200

    def available(self) -> bool:
        return True

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) == 0:
            return np.zeros(self.dim, dtype=np.float32)

        win_len = int(_WINDOW_DURATION * sr)
        hop_len = int(_WINDOW_HOP * sr)
        min_seg = max(sr // 4, 512)

        windows = []
        start = 0
        while start < len(y):
            seg = y[start: start + win_len]
            if len(seg) >= min_seg:
                windows.append(_extract_mfcc_window(seg, sr))
            start += hop_len

        if not windows:
            windows.append(_extract_mfcc_window(y, sr))

        vec = (
            _window_coeff_stats([w["mfcc"] for w in windows])
            + _window_coeff_stats([w["delta_mfcc"] for w in windows])
            + _window_coeff_stats([w["contrast"] for w in windows])
            + _stats4([w["centroid"] for w in windows])
            + _stats4([w["bandwidth"] for w in windows])
            + _stats4([w["rolloff"] for w in windows])
            + _stats4([w["zcr"] for w in windows])
            + _stats4([w["rms"] for w in windows])
        )
        return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# SpectralExtractor  (512-dim mel-spectrogram statistics)
# ---------------------------------------------------------------------------

_N_MELS = 128


class SpectralExtractor:
    """
    512-dimensional mel-spectrogram feature vector.

    Computes a 128-band mel spectrogram and summarises each band with
    mean/std/10th-pct/90th-pct across time frames (128 × 4 = 512 dims).
    Captures spectral shape without temporal fine-structure — complementary
    to MFCCs for distinguishing broad tonal categories.
    """

    name = "spectral"
    dim = 512

    def available(self) -> bool:
        return True

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) == 0:
            return np.zeros(self.dim, dtype=np.float32)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, frames)

        vec: list[float] = []
        for band in mel_db:
            vec.extend(_stats4(band))
        return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# PANNsExtractor  (2048-dim CNN14 embeddings — optional dep: panns-inference)
# ---------------------------------------------------------------------------

_PANNS_SR = 32_000  # CNN14 expects 32 kHz


class PANNsExtractor:
    """
    2048-dimensional CNN14 audio embedding from PANNs (Pretrained Audio Neural Networks).

    Uses the CNN14 model trained on AudioSet (Kong et al. 2020).  The model
    weights (~600 MB) are downloaded to ~/panns_data/ on first use.

    Requires: pip install panns-inference torch

    Note: CNN14 was trained on general AudioSet sounds. It may capture useful
    low-level acoustic structure for marine recordings, but is not specifically
    tuned for underwater bioacoustics.
    """

    name = "panns"
    dim = 2048

    def __init__(self) -> None:
        self._model: Optional[object] = None

    def available(self) -> bool:
        try:
            import panns_inference  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        from panns_inference import AudioTagging
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading PANNs CNN14 model (may download ~600 MB on first run)…")
        self._model = AudioTagging(checkpoint_path=None, device=device)

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        if not self.available():
            raise RuntimeError(
                "PANNs extractor requires 'panns-inference' and 'torch'. "
                "Install with: pip install panns-inference torch"
            )
        if self._model is None:
            self._load()

        # Resample to 32 kHz if needed
        if sr != _PANNS_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=_PANNS_SR)

        # CNN14 expects (batch, samples) as float32
        audio = y.astype(np.float32)[np.newaxis, :]

        result = self._model.inference(audio)
        # panns-inference returns (clipwise_output, embedding) tuple
        _, embedding = result  # embedding shape: (1, 2048)
        return np.array(embedding[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "mfcc": MFCCExtractor,
    "spectral": SpectralExtractor,
    "panns": PANNsExtractor,
}


def list_extractors() -> list[dict]:
    """Return info dicts for all registered extractors."""
    rows = []
    for key, cls in _REGISTRY.items():
        inst = cls()
        rows.append({
            "name": key,
            "dim": inst.dim,
            "available": inst.available(),
            "description": (cls.__doc__ or "").strip().splitlines()[0],
        })
    return rows


def get_extractor(name: str) -> FeatureExtractor:
    """Return an instantiated extractor by name, raising ValueError for unknown names."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown extractor '{name}'. Available: {', '.join(_REGISTRY)}"
        )
    inst = _REGISTRY[name]()
    if not inst.available():
        raise RuntimeError(
            f"Extractor '{name}' is not available. "
            "Install its optional dependencies (see the class docstring)."
        )
    return inst
