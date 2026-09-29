"""Train a reproducible churn model. Run `python telco_churn.py --help`."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

PROJECT_DIR = Path(__file__).resolve().parent
# The workspace's embedded Python omits the script directory from its import path.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from threadpoolctl import threadpool_limits

from churn.data import FEATURES, load_training_data
from churn.modeling import candidate_models, choose_threshold, evaluate, make_bundle
from churn.reporting import save_eda, save_evaluation

LOGGER = logging.getLogger(__name__)


def split_data(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple:
    """Use disjoint 60% training, 20% validation, and 20% test partitions."""
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_development, y_development, test_size=0.25,
        stratify=y_development, random_state=seed,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def environment_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "matplotlib",
                "scipy", "threadpoolctl")
    return {package: version(package) for package in packages}


def parameter_metadata(pipeline) -> dict:
    """Describe nonfinite estimator defaults explicitly instead of writing invalid JSON."""
    return {
        key: {"special_float": str(value)}
        if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in pipeline.named_steps["model"].get_params(deep=False).items()
    }


def train(data_path: Path, output_dir: Path, *, seed: int = 42,
          folds: int = 5, jobs: int = 2) -> Path:
    """Select by training CV, tune on validation, then assess the fixed model on test."""
    if folds < 2 or jobs < 1 or not 0 <= seed < 2**32:
        raise ValueError("Use folds >= 2, jobs >= 1, and a seed between 0 and 2**32 - 1.")
    data_path = data_path.resolve()
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    X, y = load_training_data(data_path)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, seed)
    if y_train.value_counts().min() < folds:
        raise ValueError("There are fewer training examples in one class than CV folds.")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != dataset_hash:
        raise ValueError("Dataset changed while it was being loaded. Retry with a stable file.")
    LOGGER.info("Loaded %d customers. Split: %d train / %d validation / %d test.",
                len(X), len(X_train), len(X_valid), len(X_test))

    candidates = candidate_models(seed, jobs)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    comparison = []
    # Sequential folds plus a thread cap avoid CPU oversubscription on a personal machine.
    with threadpool_limits(limits=jobs):
        for name, pipeline in candidates.items():
            LOGGER.info("Cross-validating %s (%d folds)...", name, folds)
            scores = cross_val_score(
                pipeline, X_train, y_train, scoring="average_precision",
                cv=cv, n_jobs=1, error_score="raise",
            )
            comparison.append({
                "model": name, "cv_average_precision_mean": float(scores.mean()),
                "cv_average_precision_std": float(scores.std(ddof=1)),
                "fold_scores": scores.tolist(),
            })
            LOGGER.info("%s: mean AP %.3f, fold SD %.3f", name, scores.mean(), scores.std(ddof=1))

        # Stable sorting preserves the simpler model's priority when scores tie exactly.
        comparison.sort(key=lambda row: row["cv_average_precision_mean"], reverse=True)
        selected_name = comparison[0]["model"]
        selected = clone(candidates[selected_name]).fit(X_train, y_train)
        validation_scores = selected.predict_proba(X_valid)[:, 1]
        threshold, _ = choose_threshold(y_valid, validation_scores)
        LOGGER.info("Selected %s using training CV; validation F1 cutoff = %.4f.",
                    selected_name, threshold)

        # No refit after tuning: refitting could shift scores and invalidate this cutoff.
        test_scores = selected.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_scores, threshold)
        baseline = clone(candidates["Prior baseline"]).fit(X_train, y_train)
        baseline_metrics = evaluate(y_test, baseline.predict_proba(X_test)[:, 1], 0.5)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    metadata = {
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"filename": data_path.name, "sha256": dataset_hash, "rows": len(X)},
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": environment_versions(),
        "seed": seed, "cv_folds": folds, "jobs": jobs,
        "model_selection": "training_cv_average_precision_mean",
        "threshold_selection": "validation_f1; ties choose highest cutoff",
        "selected_model": selected_name, "threshold": threshold,
        "feature_columns": list(FEATURES),
        "splits": {
            name: {"rows": len(labels), "churn_rate": float(labels.mean())}
            for name, labels in [("train", y_train), ("validation", y_valid), ("test", y_test)]
        },
        "candidate_parameters": {
            name: parameter_metadata(pipeline)
            for name, pipeline in candidates.items()
        },
    }
    report = {
        "metadata": metadata, "cross_validation": comparison,
        "validation": evaluate(y_valid, validation_scores, threshold),
        "test": test_metrics, "test_prior_baseline_at_0_5": baseline_metrics,
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / run_id
    # Publish the directory only after every artifact has been written successfully.
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=output_dir) as staging:
        stage = Path(staging)
        (stage / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8",
        )
        (stage / "split_manifest.json").write_text(json.dumps({
            "dataset_sha256": dataset_hash,
            "note": "Zero-based CSV data row positions, excluding the header.",
            "train": X_train.index.tolist(), "validation": X_valid.index.tolist(),
            "test": X_test.index.tolist(),
        }, indent=2), encoding="utf-8")
        pd.DataFrame([{key: value for key, value in row.items() if key != "fold_scores"}
                      for row in comparison]).to_csv(stage / "model_comparison.csv", index=False)
        joblib.dump(make_bundle(selected, threshold, selected_name, metadata),
                    stage / "churn_model.joblib", compress=3)
        save_eda(X_train, y_train, stage)
        save_evaluation(y_test, test_scores, threshold, selected_name, stage)
        stage.rename(final_dir)

    LOGGER.info("Test: AP %.3f | precision %.3f | recall %.3f | F1 %.3f",
                test_metrics["average_precision"], test_metrics["precision"],
                test_metrics["recall"], test_metrics["f1"])
    LOGGER.info("Saved completed run: %s", final_dir)
    return final_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "Telco-Customer-Churn.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Stratified training CV folds (default: 5).")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum model threads (default: 2).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train(args.data, args.output, seed=args.seed, folds=args.folds, jobs=args.jobs)
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
