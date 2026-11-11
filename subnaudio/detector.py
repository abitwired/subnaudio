"""Energy-based audio event detection."""

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

from .models import AudioEvent


def load_audio(file_path: Path, sr: int = None) -> Tuple[np.ndarray, int]:
    """Load a WAV file. Returns (samples, sample_rate)."""
    y, sr = librosa.load(str(file_path), sr=sr, mono=True)
    return y, sr


def detect_events(
    y: np.ndarray,
    sr: int,
    file_path: Path,
    top_db: float = 40.0,
    min_duration: float = 0.1,
    merge_gap: float = 0.5,
) -> List[AudioEvent]:
    """
    Detect non-silent audio events using energy thresholding.

    Args:
        y: Audio samples.
        sr: Sample rate.
        file_path: Source file (stored on each event for reference).
        top_db: Silence threshold — frames more than this many dB below the
                peak are considered silent.
        min_duration: Discard events shorter than this (seconds).
        merge_gap: Merge consecutive events separated by less than this (seconds).
    """
    intervals = librosa.effects.split(y, top_db=top_db)

    raw: List[Tuple[float, float]] = []
    for start_sample, end_sample in intervals:
        start = start_sample / sr
        end = end_sample / sr
        if end - start >= min_duration:
            raw.append((start, end))

    # Merge events that are close together
    merged: List[Tuple[float, float]] = []
    for start, end in raw:
        if merged and (start - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [
        AudioEvent(
            file_path=file_path,
            start_time=s,
            end_time=e,
            duration=e - s,
        )
        for s, e in merged
    ]
