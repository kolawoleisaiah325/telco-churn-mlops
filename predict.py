"""Predict churn for raw customer CSV rows using a saved model bundle."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import joblib

from churn.data import read_csv
from churn.modeling import predict_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="This project's churn_model.joblib.")
    parser.add_argument("--input", type=Path, required=True, help="Customer CSV; Churn is optional.")
    parser.add_argument("--output", type=Path, required=True, help="New CSV to create; must not exist.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}. Choose a new output filename.")
        # Load only bundles produced by this project or another trusted source.
        bundle = joblib.load(args.model)
        result = predict_frame(bundle, read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, mode="x")
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            EOFError, pickle.UnpicklingError) as exc:
        detail = str(exc) or "The saved model is empty or truncated. Use a completed training run."
        logging.error("Prediction failed: %s", detail)
        return 1
    logging.info("Saved %d customer predictions to %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
