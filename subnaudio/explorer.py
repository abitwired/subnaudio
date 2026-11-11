"""
Subnaudio Acoustic Event Explorer — Streamlit app.

Launch via the TUI's [5] Launch Explorer option, or directly:
    streamlit run subnaudio/explorer.py
"""

import io
import os
import sys
from pathlib import Path

# Allow running directly (streamlit run subnaudio/explorer.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from subnaudio.annotations import AnnotationStore, ANNOTATION_LABELS, REVIEW_STATUSES
from subnaudio.models import AudioEvent
from subnaudio.session import load_session
from subnaudio.report import _render_spectrogram

try:
    import umap as umap_lib
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# ── Constants ─────────────────────────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#22d3ee", "#4ade80", "#facc15", "#f472b6", "#fb923c",
    "#a78bfa", "#34d399", "#60a5fa", "#f87171", "#e879f9",
]
NOISE_COLOUR = "#94a3b8"
UNCLASSIFIED_COLOUR = "#64748b"

SESSION_PATH = Path(os.environ.get("SUBNAUDIO_SESSION", "subnaudio_session.json"))
ANNOTATIONS_PATH = SESSION_PATH.parent / (SESSION_PATH.stem + "_annotations.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _event_colour(cluster_id):
    if cluster_id is None:
        return UNCLASSIFIED_COLOUR
    if cluster_id == -1:
        return NOISE_COLOUR
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


@st.cache_data(show_spinner=False)
def _load_session(path: str):
    return load_session(Path(path))


@st.cache_data(show_spinner=False)
def _compute_projection(features_json: str, method: str):
    """Compute 2-D projection; keyed by JSON repr of features so cache invalidates on new data."""
    import json
    X = np.array(json.loads(features_json), dtype=np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    if method == "UMAP" and _HAS_UMAP:
        n_neighbors = min(15, max(2, len(X) - 1))
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, verbose=False)
        return reducer.fit_transform(X_scaled)
    else:
        n_comp = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
        result = PCA(n_components=n_comp).fit_transform(X_scaled)
        if result.shape[1] == 1:
            result = np.column_stack([result, np.zeros(len(result))])
        return result


@st.cache_data(show_spinner=False)
def _extract_audio_bytes(file_path: str, start: float, end: float) -> bytes:
    """Extract an event segment and return WAV bytes."""
    y, sr = librosa.load(file_path, sr=None, mono=True, offset=start, duration=end - start)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


@st.cache_data(show_spinner=False)
def _make_spectrogram(file_path: str, start: float, end: float, accent: str) -> bytes:
    y, sr = librosa.load(file_path, sr=None, mono=True, offset=start, duration=end - start)
    return _render_spectrogram(y, sr, title=Path(file_path).name, accent=accent,
                               fig_w=6.0, fig_h=3.0, return_bytes=True)


@st.cache_data(show_spinner=False)
def _make_waveform_fig(file_path: str, start: float, end: float, accent: str):
    y, sr = librosa.load(file_path, sr=None, mono=True, offset=start, duration=end - start)
    times = np.linspace(start, end, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=y, mode="lines", line=dict(color=accent, width=0.8),
        hovertemplate="t=%{x:.3f}s<br>amp=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=40, r=10, t=10, b=30),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(title="Time (s)", color="#94a3b8", gridcolor="#1e293b", tickfont=dict(size=9)),
        yaxis=dict(title="Amp", color="#94a3b8", gridcolor="#1e293b", tickfont=dict(size=9)),
        showlegend=False,
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Subnaudio Explorer",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    div[data-testid="stMetric"] { background:#1e293b; border-radius:8px; padding:0.5rem 1rem; }
    div[data-testid="stMetric"] label { color: #64748b !important; font-size:0.7rem; }
    .stForm { background:#1e293b; border-radius:8px; padding:1rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Session loading ───────────────────────────────────────────────────────
    if not SESSION_PATH.exists():
        st.markdown("# 🌊 Subnaudio Explorer")
        st.error(f"Session file not found: **{SESSION_PATH.resolve()}**")
        st.info("Run the Subnaudio TUI → **[2] Analyze** to create a session, then **[5] Launch Explorer**.")
        return

    with st.spinner("Loading session…"):
        events = _load_session(str(SESSION_PATH))

    store = AnnotationStore(ANNOTATIONS_PATH)

    if not events:
        st.warning("Session contains no events.")
        return

    # ── Session state ─────────────────────────────────────────────────────────
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = events[0].event_id

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🌊 Explorer")
        st.caption(f"`{SESSION_PATH.name}`")
        st.divider()

        # Stats
        n_annotated = store.count()
        st.metric("Events", len(events))
        c1, c2 = st.columns(2)
        unique_clusters = sorted({e.cluster_id for e in events if e.cluster_id is not None and e.cluster_id >= 0})
        c1.metric("Clusters", len(unique_clusters))
        c2.metric("Annotated", f"{n_annotated}/{len(events)}")

        if n_annotated > 0:
            st.progress(n_annotated / len(events))

        st.divider()

        # Projection method
        proj_method = st.selectbox(
            "Projection",
            ["UMAP", "PCA"] if _HAS_UMAP else ["PCA"],
            help="Dimensionality reduction for the scatter plot.",
        )

        # Filters
        st.markdown("**Filters**")
        all_cluster_labels = sorted({e.label for e in events})
        selected_cluster_labels = st.multiselect("Cluster", all_cluster_labels, default=all_cluster_labels)

        all_files = sorted({e.file_path.name for e in events})
        selected_files = st.multiselect("Source file", all_files, default=all_files)

        ann_filter = st.multiselect(
            "Annotation status",
            ["unreviewed", "annotated"],
            default=["unreviewed", "annotated"],
        )

        st.divider()

        # Annotation counts
        label_counts = store.label_counts()
        if label_counts:
            st.markdown("**Annotation breakdown**")
            for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
                st.caption(f"`{lbl}` × {cnt}")

    # ── Filter events ─────────────────────────────────────────────────────────
    def _pass_filter(e: AudioEvent) -> bool:
        if e.label not in selected_cluster_labels:
            return False
        if e.file_path.name not in selected_files:
            return False
        is_annotated = store.get(e.event_id) is not None
        if "annotated" not in ann_filter and is_annotated:
            return False
        if "unreviewed" not in ann_filter and not is_annotated:
            return False
        return True

    visible = [e for e in events if _pass_filter(e)]

    if not visible:
        st.warning("No events match the current filters.")
        return

    # ── Compute projection ────────────────────────────────────────────────────
    with_features = [e for e in visible if e.features is not None]
    if len(with_features) < 2:
        st.warning("Need at least 2 events with features to build projection.")
        return

    import json as _json
    features_key = _json.dumps([e.features.tolist() for e in with_features])
    with st.spinner(f"Computing {proj_method} projection…"):
        coords = _compute_projection(features_key, proj_method)

    # ── Build scatter dataframe ───────────────────────────────────────────────
    rows = []
    for i, event in enumerate(with_features):
        ann = store.get(event.event_id)
        rows.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "event_id": event.event_id,
            "cluster": event.label,
            "file": event.file_path.name,
            "duration": f"{event.duration:.2f}s",
            "annotation": ann["annotation"] if ann else "—",
            "status": ann["review_status"] if ann else "unreviewed",
            "colour": _event_colour(event.cluster_id),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🌊 Acoustic Event Explorer")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig = px.scatter(
        df,
        x="x", y="y",
        color="cluster",
        color_discrete_map={row["cluster"]: row["colour"] for _, row in df.iterrows()},
        custom_data=["event_id"],
        hover_data={"x": False, "y": False, "file": True, "duration": True,
                    "cluster": True, "annotation": True},
        labels={"x": f"{proj_method} 1", "y": f"{proj_method} 2"},
        height=420,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="#0f172a")))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=11),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        xaxis=dict(gridcolor="#334155", zeroline=False),
        yaxis=dict(gridcolor="#334155", zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # Highlight selected point
    sel = df[df["event_id"] == st.session_state.selected_id]
    if not sel.empty:
        fig.add_trace(go.Scatter(
            x=sel["x"], y=sel["y"], mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip",
        ))

    chart_state = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["points"],
        key="scatter",
    )

    if chart_state and chart_state.selection and chart_state.selection.points:
        pt = chart_state.selection.points[0]
        new_id = pt.get("customdata", [None])[0]
        if new_id is not None and new_id != st.session_state.selected_id:
            st.session_state.selected_id = new_id
            st.rerun()

    # ── Event detail panel ────────────────────────────────────────────────────
    selected_event = next((e for e in events if e.event_id == st.session_state.selected_id), events[0])
    accent = _event_colour(selected_event.cluster_id)

    st.divider()
    detail_col, ann_col = st.columns([6, 4], gap="large")

    with detail_col:
        st.markdown(f"#### Event {selected_event.event_id}  "
                    f"<span style='color:{accent};font-size:0.9rem'>{selected_event.label}</span>",
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Start", f"{selected_event.start_time:.2f}s")
        m2.metric("End", f"{selected_event.end_time:.2f}s")
        m3.metric("Duration", f"{selected_event.duration:.2f}s")
        m4.metric("File", selected_event.file_path.stem[:18])

        # Spectrogram
        with st.spinner("Rendering spectrogram…"):
            spec_bytes = _make_spectrogram(
                str(selected_event.file_path),
                selected_event.start_time,
                selected_event.end_time,
                accent,
            )
        st.image(spec_bytes, use_container_width=True)

        # Waveform
        waveform_fig = _make_waveform_fig(
            str(selected_event.file_path),
            selected_event.start_time,
            selected_event.end_time,
            accent,
        )
        st.plotly_chart(waveform_fig, use_container_width=True, config={"displayModeBar": False})

        # Audio player
        with st.spinner("Extracting audio…"):
            audio_bytes = _extract_audio_bytes(
                str(selected_event.file_path),
                selected_event.start_time,
                selected_event.end_time,
            )
        st.audio(audio_bytes, format="audio/wav")

        # Navigate events
        st.caption("Navigate events")
        nav_ids = [e.event_id for e in visible]
        cur_idx = nav_ids.index(st.session_state.selected_id) if st.session_state.selected_id in nav_ids else 0
        nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
        if nav_c1.button("◀ Prev", use_container_width=True) and cur_idx > 0:
            st.session_state.selected_id = nav_ids[cur_idx - 1]
            st.rerun()
        nav_c2.caption(f"{cur_idx + 1} / {len(nav_ids)}")
        if nav_c3.button("Next ▶", use_container_width=True) and cur_idx < len(nav_ids) - 1:
            st.session_state.selected_id = nav_ids[cur_idx + 1]
            st.rerun()

    with ann_col:
        st.markdown("#### Annotation")

        existing = store.get(selected_event.event_id)
        default_label = existing["annotation"] if existing else "unknown"
        default_status = existing["review_status"] if existing else "unreviewed"
        default_notes = existing["notes"] if existing else ""

        if existing:
            ts = existing.get("annotated_at", "")[:16].replace("T", " ")
            st.success(f"Annotated · {ts}")

        with st.form(key=f"ann_{selected_event.event_id}"):
            label = st.selectbox(
                "Label",
                ANNOTATION_LABELS,
                index=ANNOTATION_LABELS.index(default_label) if default_label in ANNOTATION_LABELS else 0,
            )
            status = st.selectbox(
                "Review status",
                REVIEW_STATUSES,
                index=REVIEW_STATUSES.index(default_status) if default_status in REVIEW_STATUSES else 0,
            )
            notes = st.text_area("Notes", value=default_notes, height=80,
                                 placeholder="e.g. clear 20Hz pulses, noisy background…")

            save_col, del_col = st.columns([3, 1])
            saved = save_col.form_submit_button("💾 Save", use_container_width=True, type="primary")
            deleted = del_col.form_submit_button("🗑", use_container_width=True)

        if saved:
            store.save(selected_event.event_id, label, status, notes)
            st.rerun()
        if deleted and existing:
            store.delete(selected_event.event_id)
            st.rerun()

        # Feature profile
        if selected_event.features is not None:
            st.markdown("#### Feature profile")
            feat = selected_event.features
            # Show first 20 MFCC means as a quick visual
            mfcc_means = feat[:20]
            mfcc_fig = go.Figure(go.Bar(
                x=[f"M{i}" for i in range(len(mfcc_means))],
                y=mfcc_means,
                marker_color=accent,
            ))
            mfcc_fig.update_layout(
                height=160, margin=dict(l=30, r=10, t=10, b=30),
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                xaxis=dict(color="#94a3b8", tickfont=dict(size=8), gridcolor="#1e293b"),
                yaxis=dict(color="#94a3b8", tickfont=dict(size=8), gridcolor="#1e293b"),
                showlegend=False,
            )
            st.plotly_chart(mfcc_fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("MFCC coefficients 0–19 (mean)")


if __name__ == "__main__":
    main()
