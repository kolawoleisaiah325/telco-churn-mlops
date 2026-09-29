"""Model construction, decision thresholds, and reusable inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TelcoFeatures

ARTIFACT_VERSION = 1


def build_pipeline(estimator: Any, *, scale_numeric: bool = False) -> Pipeline:
    """All learned imputations/encodings stay inside CV folds and saved models."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([
        ("clean", TelcoFeatures()), ("preprocess", preprocessing), ("model", estimator),
    ])


def candidate_models(seed: int, jobs: int) -> dict[str, Pipeline]:
    """A simple reference model plus three fixed, reproducible candidates."""
    return {
        "Prior baseline": build_pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            scale_numeric=True,
        ),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=jobs,
        )),
        "XGBoost": build_pipeline(XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=jobs,
        )),
    }


def choose_threshold(y: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Maximize F1 at every distinct score; break ties toward fewer alerts."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(y)
    if (scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or not len(scores)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Threshold selection needs aligned, finite scores between 0 and 1.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes (0 and 1).")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    # The curve's final precision/recall point has no threshold and is excluded above.
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best]), float(f1[best])


def evaluate(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "classification_report": classification_report(
            y, predicted, labels=[0, 1], target_names=["Stayed", "Churned"],
            output_dict=True, zero_division=0,
        ),
    }


def make_bundle(pipeline: Pipeline, threshold: float, model_name: str,
                metadata: dict[str, Any]) -> dict[str, Any]:
    """Store the cutoff with the exact fitted pipeline used for test evaluation."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": pipeline, "threshold": threshold, "model_name": model_name,
        "feature_columns": list(FEATURES), "metadata": metadata,
    }


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Score raw CSV rows using the persisted cleaning and tuned cutoff."""
    if bundle.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact version. Retrain using this project.")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold is invalid.")
    scores = bundle["pipeline"].predict_proba(frame)[:, 1]
    result = pd.DataFrame({
        "churn_score": scores,
        "predicted_churn": np.where(scores >= threshold, "Yes", "No"),
    }, index=frame.index)
    if "customerID" in frame:
        result.insert(0, "customerID", frame["customerID"])
    return result
