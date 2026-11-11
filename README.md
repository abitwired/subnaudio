# 🌊 Subnaudio

**Semi-supervised acoustic discovery platform for underwater sounds.**

Subnaudio detects and clusters acoustic events in audio recordings — whale calls, boat motors, explosions, biological choruses, and more — without requiring labels. Once clusters are found, an interactive browser-based Explorer lets you listen to each event, inspect spectrograms, and annotate discoveries with verified species or sound type labels.

---

## How it works

```
Audio files
    │
    ▼
[1] Event detection      — energy thresholding splits audio into non-silent segments
    │
    ▼
[2] Feature extraction   — 200-dim vector per event (MFCCs, delta-MFCCs, spectral
    │                       contrast, centroid, bandwidth, rolloff, ZCR, RMS)
    ▼
[3] Unsupervised clustering — agglomerative / K-Means / DBSCAN; auto-selects k
    │                          via silhouette score when unspecified
    ▼
[4] Review & annotate    — interactive Explorer with UMAP projection, audio
                            playback, spectrograms, and human annotation
```

---

## Installation

Requires **Python ≥ 3.9** on an arm64 or x86_64 machine.

```bash
# 1. Create a virtual environment (recommended — avoids architecture mismatches on Apple Silicon)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install as a command
pip install -e .
```

> **Apple Silicon (M1/M2/M3) note:** Use the Homebrew Python at `/opt/homebrew/bin/python3`
> rather than the system Python at `/Library/Frameworks/…`, which is x86_64 and will fail
> to load numpy.

---

## Running Subnaudio

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Launch the TUI
python -m subnaudio

# Or, after pip install -e .
subnaudio
```

The TUI opens in your terminal. No display required.

---

## TUI walkthrough

### Main menu

```
[1] Add audio files
[2] Analyze          ← detect events + extract features + cluster
[3] View results
[4] Generate report  ← self-contained HTML with spectrograms
[5] Launch Explorer  ← interactive browser UI with annotations
[6] Export to CSV
[7] Settings
[q] Quit
```

---

### Step 1 — Add audio files

Choose **[1]** and enter file paths. Glob patterns are supported:

```
Path: sample-sounds/whale-sounds-wav/*.wav
Path: /data/recordings/**/*.mp3
Path: /data/recordings/deployment_01.flac
```

**Supported formats:** `.wav` `.mp3` `.ogg` `.flac` `.aiff` `.aif` `.au` `.caf` `.w64` `.rf64` `.mp4` `.m4a`

---

### Step 2 — Analyze

Choose **[2]**. Subnaudio will:

1. Load each file and detect non-silent events.
2. Extract a 200-dimensional acoustic feature vector per event.
3. Cluster events using **agglomerative clustering** (default for small datasets) or K-Means / DBSCAN.
4. Auto-save a session file (`subnaudio_session.json`) for use in the Explorer.

Example output:

```
✓ Detected 11 event(s).
Classifying events…
✓ Found 2 cluster(s).
  Class 0    7
  Class 1    4

Session saved → subnaudio_session.json  (use [5] Launch Explorer)
```

---

### Step 3 — View results

Choose **[3]** for a summary table in the terminal:

```
 #  File                              Time range         Duration  Class
 1  blue-whale-…musculus.wav          0.00s – 92.97s     92.97s    Class 0
 2  killer-whale-…orca.wav            0.00s – 53.50s     53.50s    Class 1
 …
```

---

### Step 4 — Generate HTML report

Choose **[4]**. Subnaudio renders a **self-contained HTML file** (no server needed) with:

- Summary cards (events, clusters, files, outliers)
- A mel spectrogram for every detected event, grouped by cluster
- Dark-mode layout, opens in any browser

```
Output directory [subnaudio_report_20261111_104000]: 
✓ Report saved to /path/to/subnaudio_report_20261111_104000/index.html
  Open in a browser: open /path/to/…/index.html
```

---

### Step 5 — Launch Explorer

Choose **[5]** to open the **Acoustic Event Explorer** in your browser at `http://localhost:8501`.

The Explorer provides:

| Panel | What you can do |
|---|---|
| **UMAP scatter** | See all events projected to 2D; click any point to select it |
| **Mel spectrogram** | Inspect frequency content of the selected event |
| **Waveform** | Interactive amplitude plot over time |
| **Audio player** | Listen to the event in the browser |
| **Annotation form** | Assign a label, review status, and notes — saved independently of cluster IDs |
| **Sidebar filters** | Filter by cluster, source file, or annotation status |

#### Annotation labels

```
unknown              whale_vocalization    baleen_whale
blue_whale           fin_whale             humpback_whale
minke_whale          right_whale           sei_whale
toothed_whale        sperm_whale           killer_whale
pilot_whale          bottlenose_whale      dolphin
boat_motor           explosion             biological_chorus
noise                other
```

#### Review statuses

| Status | Meaning |
|---|---|
| `unreviewed` | Not yet inspected |
| `confirmed` | Positively identified |
| `uncertain` | Likely but not definitive |
| `rejected` | Incorrectly grouped / artefact |

Annotations are saved to `subnaudio_session_annotations.json` in the working directory.
Press **Ctrl+C** in the terminal to stop the Explorer and return to the TUI.

---

### Step 6 — Export to CSV

Choose **[6]** to write all events and cluster assignments to a CSV file.

| Column | Description |
|---|---|
| `event_id` | Sequential event number |
| `file` | Absolute path to source audio file |
| `start_time_s` | Event start (seconds) |
| `end_time_s` | Event end (seconds) |
| `duration_s` | Event duration (seconds) |
| `cluster_id` | Numeric cluster ID (`-1` = noise/outlier, blank = unclassified) |
| `label` | Human-readable label (`Class 0`, `Noise/Outlier`, etc.) |

---

## Settings (`[7]`)

| Setting | Default | Description |
|---|---|---|
| Silence threshold (top_db) | `40 dB` | Frames more than this many dB below the peak are treated as silence |
| Min event duration | `0.1 s` | Events shorter than this are discarded |
| Merge gap | `0.5 s` | Adjacent events closer than this are merged |
| Clustering method | `auto` | `auto` (agglomerative for N<30, K-Means otherwise), `agglomerative`, `kmeans`, `dbscan` |
| DBSCAN epsilon | `auto` | Neighbourhood radius — auto-computed via k-distance elbow method |
| DBSCAN min samples | `2` | Minimum events required to form a cluster |
| Fixed cluster count | `auto` | Force a specific k; `0` = auto-select via silhouette score |

---

## Output files

| File | Created by | Contents |
|---|---|---|
| `subnaudio_session.json` | Analyze | All events, features, and cluster assignments |
| `subnaudio_session_annotations.json` | Explorer | Human annotations, independent of cluster IDs |
| `subnaudio_results.csv` | Export | Tabular event summary |
| `subnaudio_report_*/index.html` | Generate report | Self-contained HTML with spectrograms |

---

## Architecture

```
subnaudio/
├── models.py       AudioEvent dataclass
├── detector.py     Energy-based event segmentation (librosa)
├── features.py     200-dim feature extraction (MFCCs, spectral, delta)
├── classifier.py   Agglomerative / K-Means / DBSCAN clustering
├── session.py      JSON session serialisation
├── annotations.py  JSON annotation store
├── report.py       Self-contained HTML report generator
├── explorer.py     Streamlit interactive explorer
└── main.py         Rich TUI
```

The analysis engine (`detector`, `features`, `classifier`) is fully independent of the UI layer and can be used as a Python library.

---

## License

MIT
