"""Unsupervised clustering of acoustic events."""

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .models import AudioEvent

# Use agglomerative clustering when N is below this threshold.
SMALL_N_THRESHOLD = 30

# Silhouette score below this triggers a low-quality warning.
LOW_SILHOUETTE_THRESHOLD = 0.15

# When using silhouette for k-selection, pick the *largest* k whose
# silhouette is within this fraction of the maximum.  This avoids
# the well-known bias where silhouette peaks at k=2 for richly
# structured data (e.g. PANNs embeddings).
SILHOUETTE_GOOD_ENOUGH_FRACTION = 0.70

try:
    import umap as _umap_lib
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def _auto_eps(X: np.ndarray, min_samples: int) -> float:
    k = min(min_samples, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    knn_dists = np.sort(dists[:, k])
    if len(knn_dists) < 3:
        return float(np.median(knn_dists))
    x = np.linspace(0, 1, len(knn_dists))
    y = (knn_dists - knn_dists.min()) / (knn_dists.max() - knn_dists.min() + 1e-12)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    unit = line_vec / np.linalg.norm(line_vec)
    pts = np.column_stack([x - x[0], y - y[0]])
    perp = np.abs(pts[:, 0] * unit[1] - pts[:, 1] * unit[0])
    return float(knn_dists[int(np.argmax(perp))])


def _auto_max_k(n: int) -> int:
    """Upper bound for k-search, scaled with dataset size (max 50)."""
    return max(2, min(int(np.sqrt(n)), n // 2, 50))


def _best_k_silhouette(
    X: np.ndarray,
    max_k: int,
    cluster_fn,
    sample_size: int = 1000,
) -> Tuple[int, np.ndarray, float]:
    """
    Find the best k using a "largest good-enough k" criterion.

    Rather than argmax(silhouette) — which is biased toward k=2 for
    richly structured embeddings — this returns the largest k whose
    silhouette score is at least SILHOUETTE_GOOD_ENOUGH_FRACTION of
    the global maximum.  This finds more clusters while still requiring
    each to be reasonably well-separated.
    """
    ss = min(sample_size, len(X))
    scores: list[Tuple[int, float, np.ndarray]] = []

    for k in range(2, max_k + 1):
        labels = cluster_fn(k, X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(
                X, labels,
                sample_size=ss if len(X) > ss else None,
                random_state=42,
            )
        except Exception:
            continue
        scores.append((k, score, labels))

    if not scores:
        return 2, cluster_fn(2, X), -1.0

    max_score = max(s for _, s, _ in scores)
    threshold = max_score * SILHOUETTE_GOOD_ENOUGH_FRACTION

    # Largest k still within the good-enough band
    good = [(k, s, lbl) for k, s, lbl in scores if s >= threshold]
    best_k, best_score, best_labels = max(good, key=lambda t: t[0])
    return best_k, best_labels, best_score


def _elbow_k(X: np.ndarray, max_k: int) -> Tuple[int, np.ndarray]:
    """K-Means inertia elbow fallback when silhouette scores are uniformly weak."""
    ks = list(range(2, max_k + 1))
    inertias, all_labels = [], []
    n_init = 3 if len(X) > 500 else 5
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init, max_iter=100)
        km.fit(X)
        inertias.append(km.inertia_)
        all_labels.append(km.labels_)
    if len(inertias) < 3:
        return ks[0], all_labels[0]
    arr = np.array(inertias, dtype=float)
    x = np.linspace(0, 1, len(arr))
    y = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    unit = line_vec / (np.linalg.norm(line_vec) + 1e-12)
    pts = np.column_stack([x - x[0], y - y[0]])
    perp = np.abs(pts[:, 0] * unit[1] - pts[:, 1] * unit[0])
    best_idx = int(np.argmax(perp))
    return ks[best_idx], all_labels[best_idx]


def _umap_reduce(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Reduce to n_components dimensions with UMAP for density-based clustering.

    Uses min_dist=0.0 to preserve local cluster structure rather than
    global spread — this produces tighter density peaks for HDBSCAN.
    """
    n_neighbors = min(15, max(2, len(X) - 1))
    reducer = _umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42,
        verbose=False,
        compatibility_layout=True,
    )
    return reducer.fit_transform(X)


def _cluster_hdbscan_umap(
    X: np.ndarray,
    min_cluster_size: int,
    n_umap_dims: int = 10,
) -> np.ndarray:
    """
    UMAP dimensionality reduction followed by HDBSCAN.

    This combination avoids the silhouette bias toward k=2 and finds
    natural density-based clusters without requiring a fixed k.
    UMAP at 10 dims preserves local neighbourhood structure far better
    than PCA for nonlinear embeddings (e.g. PANNs).
    """
    n_dims = min(n_umap_dims, X.shape[1], len(X) - 1)
    if _HAS_UMAP and n_dims >= 2:
        X_low = _umap_reduce(X, n_components=n_dims)
    else:
        n_dims = min(15, X.shape[1], len(X) - 1)
        X_low = PCA(n_components=max(2, n_dims), random_state=42).fit_transform(X)

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method="eom",
        copy=True,
    )
    return hdb.fit_predict(X_low)


def cluster_events(
    events: List[AudioEvent],
    method: str = "auto",
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 2,
    n_clusters: Optional[int] = None,
    hdbscan_min_cluster_size: Optional[int] = None,
    use_pca: bool = True,
    pca_components: int = 15,
) -> Tuple[List[AudioEvent], Dict[str, Any]]:
    """
    Assign a cluster_id to each event using unsupervised clustering.

    Args:
        method: 'auto' (recommended), 'hdbscan_umap', 'agglomerative',
                'kmeans', or 'dbscan'.
                'auto' uses agglomerative for N<30, hdbscan_umap otherwise
                (falls back to kmeans if umap-learn is not installed).
        hdbscan_min_cluster_size: Minimum events to form a cluster (HDBSCAN).
                None = auto-scale as max(3, min(N // 200, 20)).
                Increase this to get fewer, larger clusters.
                Decrease (min 2) for finer-grained groupings.
        n_clusters: Explicit cluster count for kmeans/agglomerative only.
    """
    ready = [e for e in events if e.features is not None]
    if len(ready) < 2:
        return events, {"error": "Need at least 2 events with features to cluster."}

    X = np.array([e.features for e in ready])
    X_scaled = StandardScaler().fit_transform(X)

    # PCA: use adaptive component count — higher for large-dim embeddings
    n_components = min(pca_components, len(ready) - 1, X_scaled.shape[1])
    if use_pca and n_components < X_scaled.shape[1]:
        adaptive = max(n_components, min(X_scaled.shape[1] // 8, len(ready) - 1, 50))
        n_components = adaptive
    if use_pca and n_components >= 2:
        X_scaled = PCA(n_components=n_components, random_state=42).fit_transform(X_scaled)

    # Resolve 'auto' method
    resolved_method = method
    if method == "auto":
        if len(ready) < SMALL_N_THRESHOLD:
            resolved_method = "agglomerative"
        elif _HAS_UMAP:
            resolved_method = "hdbscan_umap"
        else:
            resolved_method = "kmeans"

    max_k = _auto_max_k(len(ready))
    labels = None
    extra_stats: Dict[str, Any] = {}
    best_sil: Optional[float] = None

    # ── HDBSCAN + UMAP ──────────────────────────────────────────────────────
    if resolved_method == "hdbscan_umap":
        # min_cluster_size scales slowly: N//200 capped at 20.
        # N//50 (old formula) produces mcs=42 for N=2140, which
        # collapses weak-feature embeddings (e.g. MFCC) into 2 blobs.
        mcs = hdbscan_min_cluster_size or max(3, min(len(ready) // 200, 20))
        labels = _cluster_hdbscan_umap(X_scaled, min_cluster_size=mcs)
        n_noise = int(np.sum(labels == -1))
        non_noise = labels[labels >= 0]
        n_cl_found = len(np.unique(non_noise)) if len(non_noise) else 0

        # Detect degenerate output: ≤3 clusters where one dominates >75% of
        # non-noise events.  This happens when MFCC features produce a
        # uniform UMAP density with no tight sub-clusters.  Fall back to
        # K-Means + elbow which handles uniform spaces better.
        dominant_frac = 0.0
        if n_cl_found >= 1:
            counts = [int(np.sum(non_noise == l)) for l in np.unique(non_noise)]
            dominant_frac = max(counts) / max(len(non_noise), 1)

        is_degenerate = n_cl_found <= 3 and dominant_frac > 0.75
        fallback_reason = None

        if n_noise > len(ready) // 2:
            fallback_reason = "too_much_noise"
        elif is_degenerate:
            fallback_reason = f"degenerate_{n_cl_found}cl_{dominant_frac:.0%}_dominant"

        if fallback_reason:
            resolved_method = "kmeans"
            extra_stats["hdbscan_fallback"] = fallback_reason
        else:
            extra_stats["hdbscan_min_cluster_size"] = mcs
            extra_stats["umap_available"] = _HAS_UMAP

    # ── Agglomerative ───────────────────────────────────────────────────────
    if resolved_method == "agglomerative":
        if max_k < 2:
            return events, {"error": "Need at least 3 events to cluster."}
        if n_clusters is not None:
            chosen_k = n_clusters
            labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(X_scaled)
        else:
            chosen_k, labels, best_sil = _best_k_silhouette(
                X_scaled, max_k,
                lambda k, X: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
            )
        extra_stats["chosen_k"] = chosen_k

    # ── K-Means ─────────────────────────────────────────────────────────────
    elif resolved_method == "kmeans":
        if max_k < 2:
            return events, {"error": "Need at least 3 events for K-Means."}
        n_init = 3 if len(ready) > 500 else 5
        if n_clusters is not None:
            chosen_k = n_clusters
            labels = KMeans(n_clusters=chosen_k, random_state=42, n_init=n_init).fit_predict(X_scaled)
        else:
            chosen_k, labels, best_sil = _best_k_silhouette(
                X_scaled, max_k,
                lambda k, X: KMeans(n_clusters=k, random_state=42, n_init=n_init,
                                    max_iter=100).fit_predict(X),
            )
            if best_sil is not None and best_sil < LOW_SILHOUETTE_THRESHOLD:
                elbow_k, elbow_labels = _elbow_k(X_scaled, max_k)
                chosen_k, labels = elbow_k, elbow_labels
                extra_stats["k_method"] = "inertia_elbow"
            else:
                extra_stats["k_method"] = "silhouette_good_enough"
        extra_stats["chosen_k"] = chosen_k

    # ── DBSCAN ──────────────────────────────────────────────────────────────
    elif resolved_method == "dbscan":
        eps = dbscan_eps if dbscan_eps is not None else _auto_eps(X_scaled, dbscan_min_samples)
        labels = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(X_scaled)
        extra_stats["auto_eps"] = round(eps, 4)
        n_noise = int(np.sum(labels == -1))
        if n_noise > len(ready) // 2:
            return cluster_events(
                events, method="agglomerative",
                n_clusters=n_clusters, use_pca=use_pca, pca_components=pca_components,
            )

    elif resolved_method != "hdbscan_umap":
        return events, {"error": f"Unknown clustering method: {method!r}"}

    for event, label in zip(ready, labels):
        event.cluster_id = int(label)

    unique_labels = sorted(set(labels))
    n_noise = int(np.sum(labels == -1))
    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

    stats: Dict[str, Any] = {
        "method": resolved_method,
        "n_events": len(ready),
        "n_clusters": n_clusters_found,
        "n_noise": n_noise,
        "cluster_counts": {
            (f"Class {l}" if l >= 0 else "Noise/Outlier"): int(np.sum(labels == l))
            for l in unique_labels
        },
        "max_k_searched": max_k,
        "low_silhouette_warning": (
            best_sil is not None and best_sil < LOW_SILHOUETTE_THRESHOLD
        ),
        **extra_stats,
    }
    return events, stats

