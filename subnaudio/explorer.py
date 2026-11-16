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
    events, metrics, extractor, model_metadata = load_session(Path(path))
    return events, metrics, extractor, model_metadata


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


# ── Metrics rendering ─────────────────────────────────────────────────────────

def _render_metrics(metrics) -> None:
    """Render ClusterMetrics as Streamlit widgets in the sidebar."""
    from subnaudio.evaluation import ClusterMetrics

    def _gauge_colour(value: float, good_high: bool) -> str:
        """Return a CSS colour based on value direction."""
        if good_high:
            return "#4ade80" if value >= 0.5 else "#facc15" if value >= 0.0 else "#f87171"
        else:
            return "#4ade80" if value <= 1.0 else "#facc15" if value <= 2.0 else "#f87171"

    # Silhouette
    if metrics.silhouette_score is not None:
        v = metrics.silhouette_score
        normed = (v + 1) / 2  # map -1..1 → 0..1
        colour = "#4ade80" if v >= 0.5 else "#facc15" if v >= 0.2 else "#f87171"
        st.markdown(f"**Silhouette** <span style='color:{colour};font-weight:bold'>{v:.3f}</span> "
                    f"<span style='color:#64748b;font-size:0.75rem'>(−1..1)</span>",
                    unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, normed)))
    else:
        st.caption("Silhouette: n/a")

    # Stability
    if metrics.stability_score is not None:
        s = metrics.stability_score
        colour = "#4ade80" if s >= 0.80 else "#facc15" if s >= 0.60 else "#fb923c" if s >= 0.40 else "#f87171"
        label = metrics.stability_label()
        st.markdown(
            f"**Stability** <span style='color:{colour};font-weight:bold'>{s:.3f}</span> "
            f"<span style='color:#64748b;font-size:0.75rem'>({label}, "
            f"{metrics.n_stability_runs} runs)</span>",
            unsafe_allow_html=True,
        )
        st.progress(max(0.0, min(1.0, s)))
    else:
        st.caption("Stability: n/a")

    # Davies-Bouldin (lower is better)
    if metrics.davies_bouldin_score is not None:
        db = metrics.davies_bouldin_score
        colour = "#4ade80" if db <= 1.0 else "#facc15" if db <= 2.0 else "#f87171"
        st.markdown(
            f"**Davies-Bouldin** <span style='color:{colour};font-weight:bold'>{db:.3f}</span> "
            f"<span style='color:#64748b;font-size:0.75rem'>(lower=better)</span>",
            unsafe_allow_html=True,
        )

    # Calinski-Harabasz
    if metrics.calinski_harabasz_score is not None:
        ch = metrics.calinski_harabasz_score
        st.markdown(
            f"**Calinski-Harabasz** <span style='color:#22d3ee;font-weight:bold'>{ch:.1f}</span> "
            f"<span style='color:#64748b;font-size:0.75rem'>(higher=better)</span>",
            unsafe_allow_html=True,
        )

    # Noise ratio
    nr = metrics.noise_ratio
    colour = "#4ade80" if nr <= 0.1 else "#facc15" if nr <= 0.3 else "#f87171"
    st.markdown(
        f"**Noise ratio** <span style='color:{colour};font-weight:bold'>{nr:.1%}</span>",
        unsafe_allow_html=True,
    )

    # Cluster size distribution
    if metrics.cluster_sizes:
        sizes = sorted(metrics.cluster_sizes.items())
        size_str = "  ".join(f"`C{k}:{v}`" for k, v in sizes)
        st.caption(size_str)

    st.caption(
        "⚠️ Geometric metrics measure feature-space separation, "
        "not biological correctness. Use annotations for ground-truth validation."
    )


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
        events, metrics, extractor_name, model_metadata = _load_session(str(SESSION_PATH))

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
            st.progress(min(1.0, n_annotated / len(events)))

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

        # Cluster quality metrics
        if metrics is not None:
            st.divider()
            with st.expander("📊 Cluster Quality", expanded=True):
                _render_metrics(metrics)

        # Model training summary
        if model_metadata:
            st.divider()
            with st.expander("🤖 Model", expanded=True):
                acc = model_metadata.get("cv_accuracy")
                if acc is not None:
                    st.metric("CV Accuracy", f"{acc:.1%}")
                n_lab = model_metadata.get("n_labeled", 0)
                n_pred = model_metadata.get("n_predicted", 0)
                st.caption(f"Trained on **{n_lab}** labeled events")
                st.caption(f"Predicted **{n_pred}** unlabeled events")
                classes = model_metadata.get("classes", [])
                if classes:
                    st.caption("Classes: " + ", ".join(f"`{c}`" for c in classes))
                if model_metadata.get("low_data_warning"):
                    st.warning("⚠ Some classes have < 5 examples")
                skipped = model_metadata.get("classes_skipped", [])
                if skipped:
                    st.caption(f"Skipped (too few): {', '.join(skipped)}")

        st.divider()
        st.caption(f"Extractor: `{extractor_name}`")

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

    # Initialise multi-selection state
    if "bulk_selected_ids" not in st.session_state:
        st.session_state.bulk_selected_ids = []

    # Highlight single-selected point (detail panel)
    sel = df[df["event_id"] == st.session_state.selected_id]
    if not sel.empty:
        fig.add_trace(go.Scatter(
            x=sel["x"], y=sel["y"], mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip",
        ))

    # Highlight bulk-selected points
    if st.session_state.bulk_selected_ids:
        bulk_sel = df[df["event_id"].isin(st.session_state.bulk_selected_ids)]
        if not bulk_sel.empty:
            fig.add_trace(go.Scatter(
                x=bulk_sel["x"], y=bulk_sel["y"], mode="markers",
                marker=dict(size=14, color="rgba(250,204,21,0.25)",
                            line=dict(color="#facc15", width=2)),
                showlegend=False, hoverinfo="skip", name="_bulk_sel",
            ))

    # Selection hint caption
    n_bulk = len(st.session_state.get("bulk_selected_ids", []))
    if n_bulk:
        st.caption(
            f"📌 **{n_bulk} event(s) selected** (highlighted in yellow)  "
            "— use **⚡ Bulk Actions → Label selection** below to annotate them. "
            "Click a blank area on the chart to clear selection."
        )
    else:
        st.caption(
            "💡 **Tip:** Use **box select** or **lasso** (toolbar icons top-right of chart) "
            "to select multiple points for bulk annotation. "
            "Zoom in first to target a dense region, then drag to select."
        )

    chart_state = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        key="scatter",
    )

    if chart_state and chart_state.selection and chart_state.selection.points:
        pts = chart_state.selection.points

        new_ids = []
        for pt in pts:
            cd = pt.get("customdata")
            if cd is None:
                continue
            val = cd[0] if isinstance(cd, (list, tuple)) and len(cd) > 0 else cd
            if val is None:
                continue
            try:
                new_ids.append(int(val))
            except (TypeError, ValueError):
                pass

        # Deduplicate while preserving order
        seen: set = set()
        new_ids = [i for i in new_ids if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]

        if len(new_ids) == 1:
            if new_ids[0] != st.session_state.selected_id:
                st.session_state.selected_id = new_ids[0]
                st.session_state.bulk_selected_ids = []
                st.rerun()
        elif len(new_ids) > 1:
            if set(new_ids) != set(st.session_state.bulk_selected_ids):
                st.session_state.bulk_selected_ids = new_ids
                st.session_state.selected_id = new_ids[0]
                st.rerun()
    # NOTE: intentionally NOT auto-clearing on empty chart_state.
    # The user clicks "Clear selection" explicitly in the bulk panel.
    # Auto-clear was causing the selection to be wiped on every passive rerun.

    # ── Bulk Actions ──────────────────────────────────────────────────────────
    # Snapshot to a plain list so the value is stable within this render pass.
    sel_ids: list = list(st.session_state.get("bulk_selected_ids", []))
    n_bulk = len(sel_ids)

    with st.expander("⚡ Bulk Actions", expanded=n_bulk > 0):
        bulk_tab1, bulk_tab2, bulk_tab3, bulk_tab4, bulk_tab5 = st.tabs([
            "🎯 Label selection",
            "🔭 Label visible",
            "🏷️ Label by cluster",
            "🤖 Accept model predictions",
            "🗑️ Clear annotations",
        ])

        all_cluster_names = sorted(
            {e.label for e in events},
            key=lambda x: (x.startswith("Noise"), x),
        )

        # ── Tab 1: Label chart selection (box / lasso) ────────────────────────
        with bulk_tab1:
            if not sel_ids:
                st.info(
                    "No points selected yet.\n\n"
                    "**How to select:**\n"
                    "1. Use the **Box select** or **Lasso select** tools in the chart toolbar "
                    "(top-right of the scatter plot above)\n"
                    "2. Optionally **zoom in** first to focus on a dense region, "
                    "then draw your selection\n"
                    "3. Return here — the selection count appears in the chart caption above"
                )
            else:
                already_ann_sel = sum(1 for eid in sel_ids if store.get(eid) is not None)
                st.markdown(f"**{len(sel_ids)}** point(s) selected · {already_ann_sel} already annotated")

                s1, s2 = st.columns(2)
                sel_label = s1.selectbox("Label", ANNOTATION_LABELS, key="sel_label")
                sel_status = s2.selectbox("Status", REVIEW_STATUSES,
                                           index=REVIEW_STATUSES.index("confirmed"),
                                           key="sel_status")
                sel_overwrite = st.checkbox("Overwrite existing annotations", value=False,
                                             key="sel_overwrite")
                to_write_sel = len(sel_ids) - (already_ann_sel if not sel_overwrite else 0)
                st.caption(f"**{to_write_sel}** event(s) will be {'updated' if sel_overwrite else 'added'}.")

                sc1, sc2 = st.columns(2)
                if sc1.button("Apply to selection", type="primary", key="sel_apply",
                               disabled=to_write_sel == 0):
                    n = store.bulk_save(sel_ids, sel_label, sel_status, overwrite=sel_overwrite)
                    st.success(f"✓ Labeled {n} event(s) as **{sel_label}** / {sel_status}")
                    st.session_state.bulk_selected_ids = []
                    st.rerun()
                if sc2.button("Clear selection", key="sel_clear"):
                    st.session_state.bulk_selected_ids = []
                    st.rerun()

        # ── Tab 2: Label all currently visible (filtered) events ──────────────
        with bulk_tab2:
            st.markdown(
                "Apply a label to all events currently **visible in the scatter plot** "
                "(respects the cluster / file / annotation filters in the sidebar)."
            )
            vis_ids = [e.event_id for e in visible]
            already_ann_vis = sum(1 for eid in vis_ids if store.get(eid) is not None)

            v1, v2 = st.columns(2)
            vis_label = v1.selectbox("Label", ANNOTATION_LABELS, key="vis_label")
            vis_status = v2.selectbox("Status", REVIEW_STATUSES,
                                       index=REVIEW_STATUSES.index("confirmed"),
                                       key="vis_status")
            vis_overwrite = st.checkbox("Overwrite existing annotations", value=False,
                                         key="vis_overwrite")
            to_write_vis = len(vis_ids) - (already_ann_vis if not vis_overwrite else 0)
            st.caption(
                f"**{len(vis_ids)}** visible event(s) · {already_ann_vis} already annotated · "
                f"**{to_write_vis}** will be {'updated' if vis_overwrite else 'added'}"
            )

            if st.button("Apply to visible", type="primary", key="vis_apply",
                          disabled=to_write_vis == 0):
                n = store.bulk_save(vis_ids, vis_label, vis_status, overwrite=vis_overwrite)
                st.success(f"✓ Labeled {n} event(s) as **{vis_label}** / {vis_status}")
                st.rerun()

        # ── Tab 3: Label all events in a cluster ──────────────────────────────
        with bulk_tab3:
            st.markdown("Apply a label to every event in a cluster.")
            bc1, bc2 = st.columns(2)
            bulk_cluster = bc1.selectbox("Cluster", all_cluster_names, key="bulk_cluster")
            bulk_label = bc2.selectbox("Label", ANNOTATION_LABELS, key="bulk_label")

            bc3, bc4 = st.columns(2)
            bulk_status = bc3.selectbox("Status", REVIEW_STATUSES,
                                         index=REVIEW_STATUSES.index("confirmed"),
                                         key="bulk_status")
            bulk_overwrite = bc4.checkbox("Overwrite existing annotations", value=False,
                                           key="bulk_overwrite")

            cluster_events_ids = [e.event_id for e in events if e.label == bulk_cluster]
            already_ann = sum(1 for eid in cluster_events_ids if store.get(eid) is not None)
            to_label = len(cluster_events_ids) - (already_ann if not bulk_overwrite else 0)

            st.caption(
                f"Cluster has **{len(cluster_events_ids)}** events · "
                f"**{already_ann}** already annotated · "
                f"**{to_label}** will be {'updated' if bulk_overwrite else 'added'}"
            )

            if st.button("Apply label to cluster", type="primary", key="bulk_apply_cluster",
                          disabled=to_label == 0):
                n = store.bulk_save(
                    cluster_events_ids, bulk_label, bulk_status,
                    overwrite=bulk_overwrite,
                )
                st.success(f"✓ Labeled {n} event(s) as **{bulk_label}** / {bulk_status}")
                st.rerun()

        # ── Tab 4: Accept model predictions ───────────────────────────────────
        with bulk_tab4:
            has_predictions = any(e.predicted_label is not None for e in events)
            if not has_predictions:
                st.info("No model predictions found. Train a model first (option **[6]** in the TUI).")
            else:
                st.markdown(
                    "Promote model predictions above a confidence threshold to confirmed annotations. "
                    "Only events **without** an existing annotation are affected by default."
                )
                mp1, mp2 = st.columns(2)
                conf_threshold = mp1.slider(
                    "Min confidence", 0.5, 1.0, 0.80, 0.05, key="bulk_conf_threshold",
                    help="Only accept predictions with confidence ≥ this value",
                )
                model_status = mp2.selectbox(
                    "Set status to",
                    ["confirmed", "uncertain"],
                    key="bulk_model_status",
                )
                model_overwrite = st.checkbox(
                    "Overwrite existing annotations", value=False, key="bulk_model_overwrite"
                )

                from collections import Counter
                eligible = [
                    e for e in events
                    if e.predicted_label is not None
                    and (e.model_confidence or 0.0) >= conf_threshold
                    and (model_overwrite or store.get(e.event_id) is None)
                ]
                pred_counts = Counter(e.predicted_label for e in eligible)

                if eligible:
                    preview_rows = "  ".join(
                        f"`{lbl}` × {cnt}" for lbl, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])
                    )
                    st.caption(f"**{len(eligible)}** event(s) will be accepted:  {preview_rows}")
                else:
                    st.caption("No events meet the threshold.")

                if st.button("Accept predictions", type="primary", key="bulk_accept_model",
                              disabled=not eligible):
                    written = 0
                    for e in eligible:
                        lbl = e.predicted_label
                        matched_label = lbl if lbl in ANNOTATION_LABELS else "other"
                        if model_overwrite or store.get(e.event_id) is None:
                            store.save(e.event_id, matched_label, model_status,
                                       notes=f"auto-accepted (confidence {e.model_confidence:.2f})")
                            written += 1
                    st.success(f"✓ Accepted {written} prediction(s) as **{model_status}**")
                    st.rerun()

        # ── Tab 5: Clear annotations ───────────────────────────────────────────
        with bulk_tab5:
            st.markdown("Remove annotations from a cluster or from all events.")
            clr1, clr2 = st.columns(2)
            clear_scope = clr1.selectbox(
                "Scope",
                ["Specific cluster", "All events"],
                key="bulk_clear_scope",
            )
            if clear_scope == "Specific cluster":
                clear_cluster = clr2.selectbox(
                    "Cluster", all_cluster_names, key="bulk_clear_cluster"
                )
                clear_ids = [e.event_id for e in events if e.label == clear_cluster]
            else:
                clear_ids = [e.event_id for e in events]

            annotated_in_scope = sum(1 for eid in clear_ids if store.get(eid) is not None)
            st.caption(
                f"**{annotated_in_scope}** annotation(s) will be removed from {len(clear_ids)} event(s)."
            )

            if st.button("🗑️ Clear annotations", key="bulk_clear_btn",
                          disabled=annotated_in_scope == 0):
                n = store.bulk_delete(clear_ids)
                st.success(f"✓ Removed {n} annotation(s).")
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

        # Model prediction badge
        if selected_event.predicted_label is not None:
            conf = selected_event.model_confidence or 0.0
            conf_pct = f"{conf:.0%}"
            conf_colour = "#4ade80" if conf >= 0.8 else "#facc15" if conf >= 0.5 else "#f87171"
            st.markdown(
                f"<div style='margin:4px 0 8px;padding:6px 10px;border-radius:6px;"
                f"background:#1e293b;display:inline-block'>"
                f"🤖 <b>Model prediction:</b> "
                f"<span style='color:{conf_colour}'>{selected_event.predicted_label}</span>"
                f" <span style='color:#94a3b8;font-size:0.85em'>({conf_pct} confidence)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Spectrogram
        with st.spinner("Rendering spectrogram…"):
            spec_bytes = _make_spectrogram(
                str(selected_event.file_path),
                selected_event.start_time,
                selected_event.end_time,
                accent,
            )
        st.image(spec_bytes, width="stretch")

        # Waveform
        waveform_fig = _make_waveform_fig(
            str(selected_event.file_path),
            selected_event.start_time,
            selected_event.end_time,
            accent,
        )
        st.plotly_chart(waveform_fig, width="stretch", config={"displayModeBar": False})

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
        if nav_c1.button("◀ Prev", width="stretch") and cur_idx > 0:
            st.session_state.selected_id = nav_ids[cur_idx - 1]
            st.rerun()
        nav_c2.caption(f"{cur_idx + 1} / {len(nav_ids)}")
        if nav_c3.button("Next ▶", width="stretch") and cur_idx < len(nav_ids) - 1:
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
            saved = save_col.form_submit_button("💾 Save", width="stretch", type="primary")
            deleted = del_col.form_submit_button("🗑", width="stretch")

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
            st.plotly_chart(mfcc_fig, width="stretch", config={"displayModeBar": False})
            st.caption("MFCC coefficients 0–19 (mean)")


if __name__ == "__main__":
    main()
