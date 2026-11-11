# 🌊 Subnaudio

**Unsupervised ML classifier for underwater aquatic sounds.**

Subnaudio lets you drop in `.wav` files and automatically detects and groups acoustic events (whale calls, boat motors, explosions, biological choruses, etc.) — no labels required.

---

## How it works

1. **Event detection** — silence is stripped using energy thresholding; contiguous non-silent regions become *events*.
2. **Feature extraction** — each event is described by MFCCs, spectral centroid, bandwidth, rolloff, zero-crossing rate, and RMS energy (36-dimensional vector).
3. **Unsupervised clustering** — features are normalised, optionally PCA-reduced, then clustered with **DBSCAN** (default) or **K-Means** to discover sound classes without any human labels.

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Requires Python ≥ 3.9.

---

## Usage

```bash
python -m subnaudio
# or, after pip install -e .
subnaudio
```

### TUI workflow

```
[1] Add WAV files      — load one or more .wav files (glob patterns supported)
[2] Analyze            — detect events, extract features, run clustering
[3] View results       — table of all events with their assigned class
[4] Export to CSV      — save results for downstream analysis
[5] Settings           — tune detection threshold, clustering method, etc.
[q] Quit
```

---

## Settings

| Setting | Default | Description |
|---|---|---|
| Silence threshold (top_db) | 40 dB | Frames more than this many dB below the peak are silent |
| Min event duration | 0.1 s | Discard events shorter than this |
| Merge gap | 0.5 s | Merge events separated by less than this |
| Clustering method | `dbscan` | `dbscan` or `kmeans` |
| DBSCAN epsilon | 1.5 | Neighbourhood radius in normalised feature space |
| DBSCAN min samples | 2 | Minimum events per cluster |
| K-Means clusters | auto | Number of clusters (0 = auto via silhouette score) |

---

## Output CSV columns

| Column | Description |
|---|---|
| `event_id` | Sequential event number |
| `file` | Source WAV file path |
| `start_time_s` | Event start (seconds) |
| `end_time_s` | Event end (seconds) |
| `duration_s` | Event duration (seconds) |
| `cluster_id` | Numeric cluster ID (`-1` = noise/outlier) |
| `label` | Human-readable label (`Class 0`, `Noise/Outlier`, etc.) |

---

## License

MIT
