"""Unsupervised clustering of acoustic events."""

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .models import AudioEvent

# Use agglomerative clustering when N is below this threshold — it handles
# small datasets and non-spherical clusters far better than DBSCAN/K-Means.
SMALL_N_THRESHOLD = 30


def _auto_eps(X: np.ndarray, min_samples: int) -> float:
    """Estimate DBSCAN epsilon via the k-distance elbow method."""
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


def _best_k_silhouette(X: np.ndarray, max_k: int, cluster_fn) -> Tuple[int, np.ndarray]:
    """Find the k (2..max_k) with the highest silhouette score."""
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, max_k + 1):
        labels = cluster_fn(k, X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
    return best_k, best_labels


def cluster_events(
    events: List[AudioEvent],
    method: str = "auto",
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 2,
    n_clusters: Optional[int] = None,
    use_pca: bool = True,
    pca_components: int = 15,
) -> Tuple[List[AudioEvent], Dict[str, Any]]:
    """
    Assign a cluster_id to each event using unsupervised clustering.

    Args:
        method: 'auto' (recommended), 'agglomerative', 'kmeans', or 'dbscan'.
                'auto' picks agglomerative for N<30, kmeans otherwise.
        dbscan_eps: DBSCAN neighbourhood radius (None = auto from data).
        dbscan_min_samples: DBSCAN minimum cluster size.
        n_clusters: Explicit cluster count for kmeans/agglomerative
                    (None = auto via silhouette score).
        use_pca: Reduce dimensionality before clustering.
        pca_components: Max PCA components to retain.
    """
    ready = [e for e in events if e.features is not None]
    if len(ready) < 2:
        return events, {"error": "Need at least 2 events with features to cluster."}

    X = np.array([e.features for e in ready])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(pca_components, len(ready) - 1, X_scaled.shape[1])
    if use_pca and n_components >= 2:
        X_scaled = PCA(n_components=n_components).fit_transform(X_scaled)

    # Resolve 'auto' method
    resolved_method = method
    if method == "auto":
        resolved_method = "agglomerative" if len(ready) < SMALL_N_THRESHOLD else "kmeans"

    max_k = min(10, len(ready) - 1)
    labels = None
    extra_stats: Dict[str, Any] = {}

    if resolved_method == "agglomerative":
        if max_k < 2:
            return events, {"error": "Need at least 3 events to cluster."}
        if n_clusters is not None:
            chosen_k = n_clusters
            labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(X_scaled)
        else:
            chosen_k, labels = _best_k_silhouette(
                X_scaled, max_k,
                lambda k, X: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
            )
        extra_stats["chosen_k"] = chosen_k

    elif resolved_method == "kmeans":
        if max_k < 2:
            return events, {"error": "Need at least 3 events for K-Means."}
        if n_clusters is not None:
            chosen_k = n_clusters
            labels = KMeans(n_clusters=chosen_k, random_state=42, n_init="auto").fit_predict(X_scaled)
        else:
            chosen_k, labels = _best_k_silhouette(
                X_scaled, max_k,
                lambda k, X: KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X),
            )
        extra_stats["chosen_k"] = chosen_k

    elif resolved_method == "dbscan":
        eps = dbscan_eps if dbscan_eps is not None else _auto_eps(X_scaled, dbscan_min_samples)
        labels = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(X_scaled)
        extra_stats["auto_eps"] = round(eps, 4)
        # Fall back to agglomerative if DBSCAN produces mostly noise
        n_noise = int(np.sum(labels == -1))
        if n_noise > len(ready) // 2:
            return cluster_events(
                events, method="agglomerative",
                n_clusters=n_clusters, use_pca=use_pca, pca_components=pca_components,
            )

    else:
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
        **extra_stats,
    }
    return events, stats
