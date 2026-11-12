"""Acoustic feature extraction for audio events."""

from typing import List, Callable, Optional

import librosa
import numpy as np

from .models import AudioEvent

N_MFCC = 20
N_CONTRAST_BANDS = 4   # safe for sr as low as 8000 Hz
CONTRAST_FMIN = 100.0
WINDOW_DURATION = 2.0  # seconds per analysis window
WINDOW_HOP = 1.0       # seconds between window starts


def _stats4(arr: np.ndarray) -> List[float]:
    """Return [mean, std, 10th-pct, 90th-pct] of a 1-D array."""
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 90)),
    ]


def _window_coeff_stats(wins: list) -> List[float]:
    """Given a list of 1-D per-window coefficient vectors, return 4 stats per coefficient."""
    mat = np.array(wins)   # (n_windows, n_coeffs)
    result = []
    for col in mat.T:
        result.extend(_stats4(col))
    return result


def _extract_window(y: np.ndarray, sr: int) -> dict:
    """Extract all raw feature matrices from a single audio segment."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # delta width must be odd and <= number of frames; minimum useful width is 3
    delta_width = max(3, min(9, (mfcc.shape[1] // 2) * 2 - 1))
    delta_width = delta_width if delta_width % 2 == 1 else delta_width - 1
    delta_width = max(3, delta_width)
    return {
        "mfcc": np.mean(mfcc, axis=1),
        "delta_mfcc": np.mean(librosa.feature.delta(mfcc, width=delta_width), axis=1),
        "contrast": np.mean(
            librosa.feature.spectral_contrast(
                y=y, sr=sr, n_bands=N_CONTRAST_BANDS, fmin=CONTRAST_FMIN
            ),
            axis=1,
        ),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "rms": float(np.mean(librosa.feature.rms(y=y))),
    }


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a fixed-length acoustic feature vector using overlapping 2-second windows.

    Features computed per window, then summarised across windows with
    mean, std, 10th-pct, 90th-pct (4 stats each):
      - MFCCs        (20 coeffs × 4 = 80 dims)
      - Delta-MFCCs  (20 coeffs × 4 = 80 dims)
      - Spec contrast (N_CONTRAST_BANDS+1 coeffs × 4 = 20 dims)
      - Centroid, bandwidth, rolloff, ZCR, RMS (5 × 4 = 20 dims)
    Total: 200 dims
    """
    if len(y) == 0:
        return np.zeros(200)

    win_len = int(WINDOW_DURATION * sr)
    hop_len = int(WINDOW_HOP * sr)
    min_seg = max(sr // 4, 512)   # 0.25 s or 512 samples

    windows = []
    start = 0
    while start < len(y):
        seg = y[start: start + win_len]
        if len(seg) >= min_seg:
            windows.append(_extract_window(seg, sr))
        start += hop_len

    if not windows:
        windows.append(_extract_window(y, sr))

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


def extract_event_features(
    events: List[AudioEvent],
    y: np.ndarray,
    sr: int,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[AudioEvent]:
    """Populate the `features` field on each AudioEvent in-place."""
    for i, event in enumerate(events):
        start = int(event.start_time * sr)
        end = int(event.end_time * sr)
        segment = y[start:end]
        event.features = extract_features(segment, sr)
        if progress:
            progress(i + 1, len(events))
    return events
