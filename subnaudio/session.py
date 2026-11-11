"""Session serialisation — save and load analysis results to/from JSON."""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from .models import AudioEvent


def save_session(events: List[AudioEvent], path: Path) -> None:
    """Serialise a list of AudioEvents (with features + cluster IDs) to JSON."""
    data = {
        "version": "1",
        "saved_at": datetime.now().isoformat(),
        "events": [
            {
                "event_id": i + 1,
                "file_path": str(e.file_path),
                "start_time": e.start_time,
                "end_time": e.end_time,
                "duration": e.duration,
                "cluster_id": e.cluster_id,
                "features": e.features.tolist() if e.features is not None else None,
            }
            for i, e in enumerate(events)
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(path: Path) -> List[AudioEvent]:
    """Deserialise AudioEvents from a JSON session file."""
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
        )
        if item.get("features") is not None:
            e.features = np.array(item["features"], dtype=np.float32)
        events.append(e)
    return events
