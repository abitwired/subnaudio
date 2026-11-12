"""Energy-based audio event detection with onset-based sub-segmentation."""

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

from .models import AudioEvent


def load_audio(file_path: Path, sr: int = None) -> Tuple[np.ndarray, int]:
    """Load an audio file. Returns (samples, sample_rate)."""
    y, sr = librosa.load(str(file_path), sr=sr, mono=True)
    return y, sr


def _split_by_onsets(
    y: np.ndarray,
    sr: int,
    offset: float,
    min_duration: float,
    max_duration: float,
    onset_delta: float = 0.07,
) -> List[Tuple[float, float]]:
    """
    Split a single continuous audio segment into sub-segments using onset detection.

    Falls back to fixed-width windows if onset detection produces too few boundaries.

    Args:
        y: Audio samples for this segment.
        sr: Sample rate.
        offset: Absolute start time of this segment in the source file (seconds).
        min_duration: Minimum sub-segment length.
        max_duration: Maximum sub-segment length (segments longer than this are split).
        onset_delta: Sensitivity for onset detection (higher = fewer onsets).

    Returns:
        List of (start_time, end_time) tuples in absolute source-file seconds.
    """
    total = len(y) / sr
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, units="time",
        hop_length=512, backtrack=True, delta=onset_delta,
    )

    # Build boundaries: 0, all onsets, total duration
    raw_bounds = np.unique(np.concatenate([[0.0], onsets, [total]]))

    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(raw_bounds) - 1:
        start = raw_bounds[i]
        end = raw_bounds[i + 1]

        # Merge forward until we reach min_duration
        while end - start < min_duration and i + 1 < len(raw_bounds) - 1:
            i += 1
            end = raw_bounds[i + 1]

        # Chop into max_duration slices
        while end - start > max_duration:
            segments.append((offset + start, offset + start + max_duration))
            start += max_duration

        if end - start >= min_duration:
            segments.append((offset + start, offset + end))
        i += 1

    # Fallback: fixed windows if nothing useful came out
    if not segments:
        t = 0.0
        while t < total:
            seg_end = min(t + max_duration, total)
            if seg_end - t >= min_duration:
                segments.append((offset + t, offset + seg_end))
            t += max_duration

    return segments


def detect_events(
    y: np.ndarray,
    sr: int,
    file_path: Path,
    top_db: float = 40.0,
    min_duration: float = 0.5,
    merge_gap: float = 0.5,
    max_duration: float = 10.0,
) -> List[AudioEvent]:
    """
    Detect acoustic events in an audio file.

    Two-stage strategy:
      1. Silence-based splitting (librosa.effects.split) to find non-silent
         regions — works well for recordings with clear silence gaps.
      2. Onset-based sub-segmentation — any region longer than ``max_duration``
         is further split using onset detection so that distinct calls within a
         continuous segment become separate events.

    Args:
        y: Audio samples.
        sr: Sample rate.
        file_path: Source file path (stored on each event).
        top_db: Silence threshold in dB below peak.
        min_duration: Discard events shorter than this (seconds).
        merge_gap: Merge consecutive events separated by less than this (seconds).
        max_duration: Split events longer than this using onset detection (seconds).

    Returns:
        List of AudioEvent objects.
    """
    total_duration = len(y) / sr

    # ── Stage 1: silence-based split ────────────────────────────────────────
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
            merged.append([start, end])

    # Fallback: treat entire file as one region if silence split found nothing
    if not merged:
        merged = [[0.0, total_duration]]

    # ── Stage 2: onset-based sub-segmentation for long segments ─────────────
    final_segments: List[Tuple[float, float]] = []
    for start, end in merged:
        dur = end - start
        if dur <= max_duration:
            if dur >= min_duration:
                final_segments.append((start, end))
        else:
            # Extract this region and sub-segment it
            seg_y = y[int(start * sr): int(end * sr)]
            sub_segs = _split_by_onsets(
                seg_y, sr,
                offset=start,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            final_segments.extend(sub_segs)

    return [
        AudioEvent(
            file_path=file_path,
            start_time=s,
            end_time=e,
            duration=e - s,
        )
        for s, e in final_segments
    ]
