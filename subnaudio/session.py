"""Session serialisation — save and load analysis results to/from JSON."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import AudioEvent


def save_session(
    events: List[AudioEvent],
    path: Path,
    metrics=None,  # Optional[ClusterMetrics]
    extractor_name: Optional[str] = None,
    model_metadata: Optional[dict] = None,
) -> None:
    """Serialise a list of AudioEvents (with features + cluster IDs) to JSON.

    Also assigns event_id to each event in-place so callers can use them
    immediately (e.g. for annotation matching in the trainer).
    """
    serialised = []
    for i, e in enumerate(events):
        e.event_id = i + 1  # assign canonical ID in-place
        serialised.append({
            "event_id": e.event_id,
            "file_path": str(e.file_path),
            "start_time": e.start_time,
            "end_time": e.end_time,
            "duration": e.duration,
            "cluster_id": e.cluster_id,
            "predicted_label": e.predicted_label,
            "model_confidence": e.model_confidence,
            "features": e.features.tolist() if e.features is not None else None,
        })

    data = {
        "version": "2",
        "saved_at": datetime.now().isoformat(),
        "extractor": extractor_name or "mfcc",
        "metrics": metrics.to_dict() if metrics is not None else None,
        "model_metadata": model_metadata,
        "events": serialised,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(path: Path):
    """Deserialise AudioEvents (and optional metrics) from a JSON session file."""
    from .evaluation import ClusterMetrics

    raw = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for item in raw["events"]:
        e = AudioEvent(
            file_path=Path(item["file_path"]),
            start_time=item["start_time"],
            end_time=item["end_time"],
            duration=item["duration"],
            cluster_id=item["cluster_id"],
            event_id=item["event_id"],
            predicted_label=item.get("predicted_label"),
            model_confidence=item.get("model_confidence"),
        )
        if item.get("features") is not None:
            e.features = np.array(item["features"], dtype=np.float32)
        events.append(e)

    metrics = None
    if raw.get("metrics"):
        try:
            metrics = ClusterMetrics.from_dict(raw["metrics"])
        except Exception:
            pass

    extractor = raw.get("extractor", "mfcc")
    model_metadata = raw.get("model_metadata")
    return events, metrics, extractor, model_metadata
