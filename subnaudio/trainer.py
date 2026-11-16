"""
Semi-supervised model training from human annotations.

Workflow
--------
1. User annotates a subset of events in the Explorer (any number of classes).
2. `train_from_annotations()` extracts the labeled subset, trains a classifier,
   cross-validates on the labeled set, then applies the model to ALL events.
3. Each event receives a `predicted_label` and `model_confidence` score.
4. The trained model is saved to disk with joblib for reuse.

Architecture
------------
- Base classifier: SVC with RBF kernel (works well with high-dim embeddings).
- Wrapper: SelfTrainingClassifier — iteratively adds high-confidence
  predictions back into the training set, exploiting the larger unlabeled pool.
  Falls back to plain SVC when N_labeled < 20 (self-training needs critical mass).
- Pre-processing: StandardScaler → PCA (up to 50 components).
- Cross-validation: StratifiedKFold(5) on labeled subset only, with graceful
  degradation to fewer folds if any class has < 5 examples.

Annotation quality filter
-------------------------
Only events with review_status in {'confirmed', 'uncertain'} are used for
training.  'rejected' and 'unreviewed' are skipped.
Events annotated as 'unknown' or 'noise' are excluded from the training set
(they would pollute class boundaries) but still receive a prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Labels that carry semantic content — excluded from training but still predicted
_SKIP_TRAIN_LABELS = {"unknown", "noise"}

# Annotation statuses that count as usable training signal
_USABLE_STATUSES = {"confirmed", "uncertain"}

# Minimum examples per class to include that class in training
MIN_EXAMPLES_PER_CLASS = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """
    Outcome of a training run.

    Contains the fitted sklearn pipeline, metadata, and per-class evaluation
    metrics from cross-validation on the labeled subset.
    """
    pipeline: object                            # fitted sklearn Pipeline
    classes: List[str]                          # human-readable class names
    n_labeled: int                              # labeled events used for training
    n_predicted: int                            # unlabeled events that received predictions
    extractor_name: str                         # e.g. "panns"
    cv_accuracy: Optional[float]               # mean cross-val accuracy on labeled set
    cv_report: Optional[Dict[str, dict]]       # per-class precision/recall/f1
    low_data_warning: bool = False             # True if any class has < 5 examples
    classes_skipped: List[str] = field(default_factory=list)  # excluded for low count

    def to_metadata_dict(self) -> dict:
        """Serialisable summary (not the full pipeline — use joblib for that)."""
        return {
            "classes": self.classes,
            "n_labeled": self.n_labeled,
            "n_predicted": self.n_predicted,
            "extractor_name": self.extractor_name,
            "cv_accuracy": self.cv_accuracy,
            "low_data_warning": self.low_data_warning,
            "classes_skipped": self.classes_skipped,
            "cv_report": self.cv_report,
        }

    @classmethod
    def metadata_from_dict(cls, d: dict) -> "ModelResult":
        """Reconstruct metadata only (pipeline will be None)."""
        return cls(
            pipeline=None,
            classes=d.get("classes", []),
            n_labeled=d.get("n_labeled", 0),
            n_predicted=d.get("n_predicted", 0),
            extractor_name=d.get("extractor_name", "unknown"),
            cv_accuracy=d.get("cv_accuracy"),
            cv_report=d.get("cv_report"),
            low_data_warning=d.get("low_data_warning", False),
            classes_skipped=d.get("classes_skipped", []),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_from_annotations(
    events: "List[AudioEvent]",
    annotations: Dict[str, dict],
    extractor_name: str = "mfcc",
    self_training: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple["ModelResult", "List[AudioEvent]"]:
    """
    Train a classifier from human annotations and apply it to all events.

    Args:
        events:          All AudioEvents (must have .features populated).
        annotations:     Dict keyed by str(event_id) from AnnotationStore.all().
        extractor_name:  Name of the feature extractor used (stored in result).
        self_training:   Use SelfTrainingClassifier to exploit unlabeled data.
        progress_cb:     Optional status callback (receives a string message).

    Returns:
        (ModelResult, updated_events) — events have .predicted_label and
        .model_confidence populated on all events with features.

    Raises:
        ValueError: if fewer than 2 usable classes with MIN_EXAMPLES_PER_CLASS.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.svm import SVC

    def _emit(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    # ── Collect labeled events ──────────────────────────────────────────────
    _emit("Collecting annotated events…")

    events_with_features = [e for e in events if e.features is not None]
    if not events_with_features:
        raise ValueError("No events have extracted features. Run Analyze first.")

    # Map event_id → annotation record
    id_to_ann: Dict[int, dict] = {}
    for eid_str, rec in annotations.items():
        status = rec.get("review_status", "unreviewed")
        label = rec.get("annotation", "unknown")
        if status in _USABLE_STATUSES and label not in _SKIP_TRAIN_LABELS:
            try:
                id_to_ann[int(eid_str)] = rec
            except ValueError:
                pass

    # ── Class frequency filter ─────────────────────────────────────────────
    label_counts: Dict[str, int] = {}
    for rec in id_to_ann.values():
        lbl = rec["annotation"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    classes_skipped = [
        lbl for lbl, cnt in label_counts.items() if cnt < MIN_EXAMPLES_PER_CLASS
    ]
    usable_classes = {
        lbl for lbl, cnt in label_counts.items() if cnt >= MIN_EXAMPLES_PER_CLASS
    }

    # Remove events whose class doesn't meet the minimum
    id_to_ann = {
        eid: rec for eid, rec in id_to_ann.items()
        if rec["annotation"] in usable_classes
    }

    if len(usable_classes) < 2:
        raise ValueError(
            f"Need at least 2 annotation classes with ≥{MIN_EXAMPLES_PER_CLASS} "
            f"confirmed examples each. Found: {dict(label_counts)}. "
            "Annotate more events in the Explorer and try again."
        )

    _emit(f"Found {len(id_to_ann)} labeled events across {len(usable_classes)} classes.")

    # ── Build feature matrix ───────────────────────────────────────────────
    _emit("Building feature matrix…")

    # Labeled
    labeled_events = [e for e in events_with_features if e.event_id in id_to_ann]
    labeled_X = np.stack([e.features for e in labeled_events]).astype(np.float32)
    labeled_y_str = np.array([id_to_ann[e.event_id]["annotation"] for e in labeled_events])

    # All (labeled + unlabeled) for self-training
    all_X = np.stack([e.features for e in events_with_features]).astype(np.float32)

    # Encode string labels → integers; unlabeled = -1
    le = LabelEncoder()
    le.fit(sorted(usable_classes))
    labeled_y = le.transform(labeled_y_str)

    all_y = np.full(len(events_with_features), -1, dtype=int)
    for i, event in enumerate(events_with_features):
        if event.event_id in id_to_ann:
            ann_label = id_to_ann[event.event_id]["annotation"]
            if ann_label in usable_classes:
                all_y[i] = le.transform([ann_label])[0]

    # ── Build sklearn pipeline ─────────────────────────────────────────────
    _emit("Building classifier pipeline…")

    dim = all_X.shape[1]
    n_pca = max(2, min(dim // 8, len(labeled_events) - 1, 50))

    base_svc = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    use_self_training = self_training and len(labeled_events) >= 20
    if use_self_training:
        classifier = SelfTrainingClassifier(
            estimator=base_svc,
            threshold=0.80,
            criterion="threshold",
            max_iter=10,
        )
    else:
        classifier = base_svc
        _emit("Note: using plain SVC (need ≥20 labeled events for self-training).")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("clf", classifier),
    ])

    # ── Cross-validate on labeled subset ───────────────────────────────────
    _emit("Cross-validating on labeled events…")

    # Determine number of folds: at least 2, at most 5, limited by min class count
    min_class_count = min(label_counts.get(c, 0) for c in usable_classes)
    n_folds = min(5, min_class_count)
    low_data_warning = min_class_count < 5

    cv_accuracy: Optional[float] = None
    cv_report: Optional[dict] = None

    if n_folds >= 2:
        try:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            # Cross-val on labeled set only, using plain SVC (SelfTraining can't be used in CV)
            cv_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=42)),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                            probability=True, class_weight="balanced", random_state=42)),
            ])
            cv_preds = cross_val_predict(cv_pipeline, labeled_X, labeled_y, cv=cv)
            correct = np.sum(cv_preds == labeled_y)
            cv_accuracy = float(correct) / len(labeled_y)

            report = classification_report(
                labeled_y, cv_preds,
                target_names=le.classes_,
                output_dict=True,
                zero_division=0,
            )
            cv_report = {
                cls_name: {
                    "precision": report[cls_name]["precision"],
                    "recall": report[cls_name]["recall"],
                    "f1": report[cls_name]["f1-score"],
                    "support": report[cls_name]["support"],
                }
                for cls_name in le.classes_
                if cls_name in report
            }
            _emit(f"Cross-validation accuracy: {cv_accuracy:.1%} ({n_folds}-fold).")
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)
            _emit(f"Cross-validation skipped: {exc}")
    else:
        _emit("Skipping cross-validation (too few examples per class).")

    # ── Fit final model on all labeled data ────────────────────────────────
    _emit("Fitting model on all labeled data…")

    if use_self_training:
        pipeline.fit(all_X, all_y)
    else:
        pipeline.fit(labeled_X, labeled_y)

    # ── Apply predictions to all events ────────────────────────────────────
    _emit("Applying model to all events…")

    proba = pipeline.predict_proba(all_X)   # (N, n_classes)
    pred_indices = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_indices)
    confidences = proba[np.arange(len(proba)), pred_indices]

    n_predicted = 0
    for i, event in enumerate(events_with_features):
        event.predicted_label = str(pred_labels[i])
        event.model_confidence = float(confidences[i])
        if all_y[i] == -1:   # was unlabeled
            n_predicted += 1

    result = ModelResult(
        pipeline=pipeline,
        classes=list(le.classes_),
        n_labeled=len(labeled_events),
        n_predicted=n_predicted,
        extractor_name=extractor_name,
        cv_accuracy=cv_accuracy,
        cv_report=cv_report,
        low_data_warning=low_data_warning,
        classes_skipped=classes_skipped,
    )

    _emit(f"Done. {n_predicted} unlabeled events predicted.")
    return result, events


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(result: ModelResult, path: Path) -> None:
    """Save the fitted sklearn pipeline to disk with joblib."""
    try:
        import joblib
    except ImportError:
        from sklearn.utils import _joblib as joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, path)


def load_model_pipeline(path: Path) -> object:
    """Load a previously saved sklearn pipeline."""
    try:
        import joblib
    except ImportError:
        from sklearn.utils import _joblib as joblib

    return joblib.load(path)
