"""Annotation persistence — store and retrieve human labels for audio events."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ANNOTATION_LABELS: List[str] = [
    "unknown",
    "whale_vocalization",
    "baleen_whale",
    "blue_whale",
    "fin_whale",
    "humpback_whale",
    "minke_whale",
    "right_whale",
    "sei_whale",
    "toothed_whale",
    "sperm_whale",
    "killer_whale",
    "pilot_whale",
    "bottlenose_whale",
    "dolphin",
    "boat_motor",
    "explosion",
    "biological_chorus",
    "noise",
    "other",
]

REVIEW_STATUSES: List[str] = [
    "unreviewed",
    "confirmed",
    "uncertain",
    "rejected",
]


class AnnotationStore:
    """
    JSON-backed store for event annotations.

    Format::

        {
            "version": "1",
            "annotations": {
                "1": {
                    "annotation": "sperm_whale",
                    "review_status": "confirmed",
                    "notes": "clear clicks at 2-4kHz",
                    "annotated_at": "2026-11-11T10:30:00"
                }
            }
        }
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                pass
        return {"version": "1", "annotations": {}}

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def save(self, event_id: int, label: str, status: str, notes: str = "") -> None:
        """Create or update the annotation for an event."""
        self._data["annotations"][str(event_id)] = {
            "annotation": label,
            "review_status": status,
            "notes": notes,
            "annotated_at": datetime.now().isoformat(),
        }
        self._flush()

    def delete(self, event_id: int) -> None:
        """Remove the annotation for an event."""
        self._data["annotations"].pop(str(event_id), None)
        self._flush()

    def get(self, event_id: int) -> Optional[Dict]:
        return self._data["annotations"].get(str(event_id))

    def all(self) -> Dict[str, Dict]:
        return self._data["annotations"]

    def count(self) -> int:
        return len(self._data["annotations"])

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ann in self._data["annotations"].values():
            lbl = ann.get("annotation", "unknown")
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts
