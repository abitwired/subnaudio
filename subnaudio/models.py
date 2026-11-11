from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class AudioEvent:
    file_path: Path
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    features: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_id: Optional[int] = None

    @property
    def label(self) -> str:
        if self.cluster_id is None:
            return "Unclassified"
        if self.cluster_id == -1:
            return "Noise/Outlier"
        return f"Class {self.cluster_id}"

    @property
    def time_range(self) -> str:
        return f"{self.start_time:.2f}s – {self.end_time:.2f}s"
