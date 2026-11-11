"""HTML cluster report with mel spectrogram visualisations."""

import base64
import io
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

from .models import AudioEvent

# ── Spectrogram rendering ────────────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#22d3ee",  # cyan
    "#4ade80",  # green
    "#facc15",  # yellow
    "#f472b6",  # pink
    "#fb923c",  # orange
    "#a78bfa",  # violet
    "#34d399",  # emerald
    "#60a5fa",  # blue
    "#f87171",  # red
    "#e879f9",  # fuchsia
]
NOISE_COLOUR = "#94a3b8"


def _cluster_colour(cluster_id: Optional[int]) -> str:
    if cluster_id is None or cluster_id == -1:
        return NOISE_COLOUR
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


def _render_spectrogram(
    y: np.ndarray,
    sr: int,
    title: str,
    accent: str = "#22d3ee",
    fig_w: float = 4.5,
    fig_h: float = 2.8,
) -> str:
    """Render a mel spectrogram and return it as a base64 PNG data URI."""
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    n_fft = min(2048, len(y) // 4 or 512)
    hop_length = n_fft // 4
    n_mels = 64

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel",
        ax=ax, cmap="magma",
    )

    ax.set_title(title, color=accent, fontsize=8, pad=4, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.yaxis.label.set_color("#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)

    fig.colorbar(img, ax=ax, format="%+2.0f dB", location="right",
                 pad=0.02).ax.tick_params(colors="#94a3b8", labelsize=5)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.3)
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── HTML generation ──────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace;
  background: #0f172a; color: #e2e8f0; padding: 2rem;
}
h1 { color: #22d3ee; font-size: 1.6rem; margin-bottom: 0.25rem; }
.subtitle { color: #64748b; font-size: 0.8rem; margin-bottom: 2rem; }
.summary-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem; margin-bottom: 2.5rem;
}
.stat-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 8px;
  padding: 1rem; text-align: center;
}
.stat-card .value { font-size: 1.8rem; font-weight: bold; color: #22d3ee; }
.stat-card .label { font-size: 0.7rem; color: #64748b; margin-top: 0.2rem; }
.cluster-section { margin-bottom: 3rem; }
.cluster-header {
  display: flex; align-items: center; gap: 0.75rem;
  margin-bottom: 1.25rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid #1e293b;
}
.cluster-dot {
  width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0;
}
.cluster-title { font-size: 1.1rem; font-weight: bold; }
.cluster-count { color: #64748b; font-size: 0.8rem; }
.spec-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem;
}
.event-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 8px;
  overflow: hidden; transition: border-color 0.2s;
}
.event-card:hover { border-color: #475569; }
.event-card img { width: 100%; display: block; }
.event-meta {
  padding: 0.5rem 0.75rem; font-size: 0.7rem; color: #94a3b8;
  border-top: 1px solid #0f172a;
}
.event-meta .filename { color: #cbd5e1; font-weight: bold; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.event-meta .timerange { margin-top: 0.1rem; }
footer { color: #334155; font-size: 0.7rem; margin-top: 3rem; text-align: center; }
"""


def _stat_card(value: str, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f'</div>'
    )


def generate_report(
    events: List[AudioEvent],
    output_dir: Path,
    progress_cb=None,
) -> Path:
    """
    Generate a self-contained HTML cluster report.

    Args:
        events: Classified AudioEvents (must have cluster_id set).
        output_dir: Directory to write the report into (will be created).
        progress_cb: Optional callable(done, total) for progress updates.

    Returns:
        Path to the generated index.html.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group events by cluster
    clusters: Dict[Optional[int], List[AudioEvent]] = {}
    for event in events:
        clusters.setdefault(event.cluster_id, []).append(event)

    # Sort: named clusters first (ascending), then noise (-1), then unclassified (None)
    def _sort_key(cid):
        if cid is None:
            return (2, 0)
        if cid == -1:
            return (1, 0)
        return (0, cid)

    sorted_cluster_ids = sorted(clusters.keys(), key=_sort_key)

    # Cache loaded audio to avoid re-loading the same file per cluster
    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

    def _get_audio(path: Path) -> Tuple[np.ndarray, int]:
        if path not in audio_cache:
            audio_cache[path] = librosa.load(str(path), sr=None, mono=True)
        return audio_cache[path]

    # Build spectrogram cards
    total_events = len(events)
    done = 0

    cluster_html_blocks = []

    for cid in sorted_cluster_ids:
        evs = clusters[cid]
        colour = _cluster_colour(cid)
        if cid is None:
            cluster_label = "Unclassified"
        elif cid == -1:
            cluster_label = "Noise / Outlier"
        else:
            cluster_label = f"Class {cid}"

        cards_html = []
        for event in evs:
            y_full, sr = _get_audio(event.file_path)
            start = int(event.start_time * sr)
            end = int(event.end_time * sr)
            segment = y_full[start:end]

            short_name = event.file_path.name
            title = f"{short_name}  {event.start_time:.1f}s–{event.end_time:.1f}s"
            data_uri = _render_spectrogram(segment, sr, title=short_name, accent=colour)

            cards_html.append(
                f'<div class="event-card">'
                f'<img src="{data_uri}" alt="spectrogram" loading="lazy">'
                f'<div class="event-meta">'
                f'<div class="filename" title="{event.file_path}">{short_name}</div>'
                f'<div class="timerange">{event.start_time:.2f}s – {event.end_time:.2f}s'
                f' &nbsp;·&nbsp; {event.duration:.2f}s</div>'
                f'</div>'
                f'</div>'
            )
            done += 1
            if progress_cb:
                progress_cb(done, total_events)

        cluster_html_blocks.append(
            f'<div class="cluster-section">'
            f'<div class="cluster-header">'
            f'<div class="cluster-dot" style="background:{colour}"></div>'
            f'<span class="cluster-title" style="color:{colour}">{cluster_label}</span>'
            f'<span class="cluster-count">{len(evs)} event(s)</span>'
            f'</div>'
            f'<div class="spec-grid">{"".join(cards_html)}</div>'
            f'</div>'
        )

    # Summary stats
    n_classes = len([c for c in sorted_cluster_ids if c is not None and c >= 0])
    n_noise = len(clusters.get(-1, []))
    unique_files = len({e.file_path for e in events})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_html = (
        f'<div class="summary-grid">'
        + _stat_card(str(total_events), "Events Detected")
        + _stat_card(str(n_classes), "Classes Found")
        + _stat_card(str(unique_files), "Audio Files")
        + _stat_card(str(n_noise), "Noise / Outliers")
        + "</div>"
    )

    html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Subnaudio Cluster Report</title>
          <style>{_CSS}</style>
        </head>
        <body>
          <h1>🌊 Subnaudio Cluster Report</h1>
          <div class="subtitle">Generated {timestamp}</div>
          {summary_html}
          {"".join(cluster_html_blocks)}
          <footer>Subnaudio · unsupervised acoustic event classifier</footer>
        </body>
        </html>
    """)

    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
