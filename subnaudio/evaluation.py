"""
Cluster evaluation — geometric quality metrics and bootstrap stability scoring.

Silhouette / Davies-Bouldin / Calinski-Harabasz measure geometric separation.
Stability measures whether the same events repeatedly group together when
clustering is re-run on slightly different (bootstrap) subsets of the data.

Important caveat
----------------
Geometric metrics reflect separation in feature space, not biological
correctness.  A high silhouette score does not guarantee biologically
meaningful clusters.  Human-verified annotations (see annotations.py)
provide a separate, ground-truth validation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score as _ch_score,
    davies_bouldin_score as _db_score,
    silhouette_score as _sil_score,
)
from sklearn.preprocessing import StandardScaler

from .models import AudioEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ClusterMetrics:
    """
    Cluster quality metrics for a single clustering result.

    Geometric metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) measure
    how well-separated the clusters are in feature space.  They do *not*
    measure biological correctness.

    stability_score measures whether the groupings are consistent across
    bootstrap re-runs of the clustering algorithm.  Values:
      ≥ 0.80  highly stable — groupings are robust
      0.60–0.79  moderately stable — broadly consistent
      0.40–0.59  somewhat unstable — treat with caution
      < 0.40  unstable — clusters may be arbitrary
    """
    silhouette_score: Optional[float]         # -1..1, higher is better
    davies_bouldin_score: Optional[float]     # ≥0, lower is better
    calinski_harabasz_score: Optional[float]  # ≥0, higher is better
    noise_ratio: float                        # fraction labelled as noise/outlier
    cluster_sizes: dict                       # {cluster_id: event_count}
    stability_score: Optional[float] = None  # 0..1 from bootstrap co-assignment
    n_stability_runs: int = 0

    def stability_label(self) -> str:
        s = self.stability_score
        if s is None:
            return "not computed"
        if s >= 0.80:
            return "highly stable"
        if s >= 0.60:
            return "moderately stable"
        if s >= 0.40:
            return "somewhat unstable"
        return "unstable"

    def to_dict(self) -> dict:
        return {
            "silhouette_score": self.silhouette_score,
            "davies_bouldin_score": self.davies_bouldin_score,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "noise_ratio": self.noise_ratio,
            "cluster_sizes": {str(k): v for k, v in self.cluster_sizes.items()},
            "stability_score": self.stability_score,
            "n_stability_runs": self.n_stability_runs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterMetrics":
        return cls(
            silhouette_score=d.get("silhouette_score"),
            davies_bouldin_score=d.get("davies_bouldin_score"),
            calinski_harabasz_score=d.get("calinski_harabasz_score"),
            noise_ratio=d.get("noise_ratio", 0.0),
            cluster_sizes={int(k): v for k, v in d.get("cluster_sizes", {}).items()},
            stability_score=d.get("stability_score"),
            n_stability_runs=d.get("n_stability_runs", 0),
        )


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------

def compute_metrics(events: List[AudioEvent], features: np.ndarray) -> ClusterMetrics:
    """
    Compute geometric cluster quality metrics for a completed clustering result.

    Args:
        events:   AudioEvent list (cluster_id must be populated).
        features: (N, D) feature matrix aligned with events.
    """
    labels = np.array([
        e.cluster_id if e.cluster_id is not None else -1
        for e in events
    ])

    noise_ratio = float(np.sum(labels == -1)) / max(len(labels), 1)

    cluster_sizes: dict[int, int] = {}
    for lbl in labels:
        if lbl >= 0:
            cluster_sizes[int(lbl)] = cluster_sizes.get(int(lbl), 0) + 1

    non_noise = labels != -1
    X = features[non_noise]
    y = labels[non_noise]

    sil, db, ch = None, None, None
    n_cls = len(np.unique(y))

    if n_cls >= 2 and len(X) >= n_cls:
        X_sc = StandardScaler().fit_transform(X)
        try:
            sil = float(_sil_score(X_sc, y))
        except Exception as exc:
            logger.debug("silhouette_score: %s", exc)
        try:
            db = float(_db_score(X_sc, y))
        except Exception as exc:
            logger.debug("davies_bouldin_score: %s", exc)
        try:
            ch = float(_ch_score(X_sc, y))
        except Exception as exc:
            logger.debug("calinski_harabasz_score: %s", exc)

    return ClusterMetrics(
        silhouette_score=sil,
        davies_bouldin_score=db,
        calinski_harabasz_score=ch,
        noise_ratio=noise_ratio,
        cluster_sizes=cluster_sizes,
    )


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def _prepare_X(features: np.ndarray, n_pca: int = 15) -> np.ndarray:
    """Scale and reduce features for bootstrap clustering."""
    X = StandardScaler().fit_transform(features)
    n_comp = min(n_pca, X.shape[0] - 1, X.shape[1])
    if n_comp >= 2:
        X = PCA(n_components=n_comp, random_state=42).fit_transform(X)
    return X.astype(np.float32)


def _bootstrap_cluster(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Cluster X into k groups.  Uses K-Means (random seed variation provides
    the perturbation needed to probe stability).
    """
    k = max(2, min(k, len(X) - 1))
    if len(X) < 30:
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=100)
    return km.fit_predict(X)


def compute_stability(
    events: List[AudioEvent],
    features: np.ndarray,
    n_runs: int = 20,
    boot_fraction: float = 0.80,
    pca_components: int = 15,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate cluster stability via bootstrap co-assignment.

    Algorithm
    ---------
    1. Run `n_runs` bootstrap iterations.  Each iteration samples
       `boot_fraction` of events without replacement.
    2. Re-cluster the sample; record which event pairs end up together.
    3. Build a co-assignment matrix: C[i,j] = fraction of runs (in which
       both i and j were sampled and assigned to a real cluster) where they
       ended up in the *same* cluster.
    4. Stability = mean C[i,j] over all pairs that belong to the same cluster
       in the original result.

    A score of 1.0 means every bootstrap run always grouped the same events
    together.  A score of 0.5 means co-cluster members are found together
    only half the time — groupings are poorly reproduced.

    Returns
    -------
    float in [0, 1], or None if fewer than 4 non-noise events.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    orig_labels = np.array([
        e.cluster_id if e.cluster_id is not None else -1
        for e in events
    ])

    non_noise_idx = np.where(orig_labels >= 0)[0]
    n_valid = len(non_noise_idx)

    if n_valid < 4:
        return None

    k = len(np.unique(orig_labels[non_noise_idx]))
    if k < 2:
        return None

    X_all = _prepare_X(features, n_pca=pca_components)
    n = len(events)
    sample_size = max(k + 1, int(boot_fraction * n_valid))

    # Co-assignment counters (use only non-noise events for efficiency)
    C = np.zeros((n_valid, n_valid), dtype=np.float32)
    N_seen = np.zeros((n_valid, n_valid), dtype=np.float32)

    # Map global idx → non_noise position
    global_to_nn = {int(gi): li for li, gi in enumerate(non_noise_idx)}

    successful_runs = 0
    for run in range(n_runs):
        # Sample from non-noise events
        sample_nn_pos = np.sort(
            rng.choice(n_valid, size=min(sample_size, n_valid), replace=False)
        )
        X_boot = X_all[non_noise_idx[sample_nn_pos]]

        try:
            seed = int(rng.integers(0, 2**31))
            run_labels = _bootstrap_cluster(X_boot, k=k, seed=seed)
        except Exception as exc:
            logger.debug("Bootstrap run %d failed: %s", run, exc)
            if progress_cb:
                progress_cb(run + 1, n_runs)
            continue

        # Vectorised co-assignment update
        ii, jj = np.triu_indices(len(sample_nn_pos), k=1)
        a_pos = sample_nn_pos[ii]  # positions in non_noise_idx
        b_pos = sample_nn_pos[jj]

        # Only update pairs where both received a real cluster label
        both_valid = (run_labels[ii] >= 0) & (run_labels[jj] >= 0)
        a_v = a_pos[both_valid]
        b_v = b_pos[both_valid]
        rl_ii = run_labels[ii][both_valid]
        rl_jj = run_labels[jj][both_valid]

        # Symmetric update of N_seen
        N_seen[a_v, b_v] += 1
        N_seen[b_v, a_v] += 1

        # Symmetric update of co-assignment matrix
        same = rl_ii == rl_jj
        C[a_v[same], b_v[same]] += 1
        C[b_v[same], a_v[same]] += 1

        successful_runs += 1
        if progress_cb:
            progress_cb(run + 1, n_runs)

    if successful_runs == 0:
        return None

    # Co-assignment rate per pair
    with np.errstate(divide="ignore", invalid="ignore"):
        coassign = np.where(N_seen > 0, C / N_seen, np.nan)

    # Find pairs that share the same original cluster (within non-noise events)
    orig_nn_labels = orig_labels[non_noise_idx]  # (n_valid,)
    ii_all, jj_all = np.triu_indices(n_valid, k=1)
    same_cluster = orig_nn_labels[ii_all] == orig_nn_labels[jj_all]

    within_coassign = coassign[ii_all[same_cluster], jj_all[same_cluster]]
    valid_within = within_coassign[~np.isnan(within_coassign)]

    if len(valid_within) == 0:
        return None

    return float(np.mean(valid_within))
