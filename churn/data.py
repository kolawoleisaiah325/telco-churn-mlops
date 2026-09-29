"""One input contract shared by training and prediction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

TARGET = "Churn"
NUMERIC_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def read_csv(path: Path) -> pd.DataFrame:
    """Reject malformed records before pandas can reinterpret them as an index."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader, [])
            if not header:
                raise ValueError("The CSV is empty.")
            if any(not column.strip() for column in header):
                raise ValueError("The CSV has an empty column name.")
            if len(header) != len(set(header)):
                raise ValueError("The CSV has duplicate column names.")
            for row in reader:
                if not row:  # Match pandas' default handling of empty lines.
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV record ending at line {reader.line_num} has {len(row)} fields; "
                        f"expected {len(header)}. Check missing or extra commas."
                    )
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV near line {reader.line_num}: {exc}") from exc
    # Only blank fields count as missing; e.g. an unexpected 'NA' label is not dropped.
    return pd.read_csv(path, encoding="utf-8-sig", dtype="string",
                       keep_default_na=False, na_values=[""])


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate without learning statistics; leave missing values for the pipeline."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Features must be a pandas DataFrame with named columns.")
    if frame.empty:
        raise ValueError("No customer rows were supplied.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate feature column names are not allowed.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    # Explicit selection prevents IDs, labels, or extra CSV fields entering the model.
    clean = frame.loc[:, list(FEATURES)].copy()
    for column in FEATURES:
        values = clean[column].astype("string").str.strip().replace("", pd.NA)
        if column in NUMERIC_FEATURES or column == "SeniorCitizen":
            numeric = pd.to_numeric(values, errors="coerce").astype(float)
            invalid = (values.notna() & numeric.isna()) | np.isinf(numeric) | (numeric < 0)
            if column == "tenure":
                invalid |= numeric.notna() & numeric.mod(1).ne(0)
            if column == "SeniorCitizen":
                invalid |= numeric.notna() & ~numeric.isin([0, 1])
            if invalid.any():
                raise ValueError(f"{column} contains {int(invalid.sum())} invalid numeric value(s).")
            if column == "SeniorCitizen":
                clean[column] = numeric.map({0.0: "0", 1.0: "1"})
            else:
                clean[column] = numeric
        else:
            # sklearn imputers use np.nan; pandas' nullable strings use pd.NA.
            clean[column] = values.astype(object).where(values.notna(), np.nan)
    return clean


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Reject ambiguous targets and repeated customer IDs before splitting."""
    data = read_csv(path)
    required = {"customerID", TARGET}
    if not required.issubset(data.columns):
        raise ValueError("Training CSV must include customerID and Churn.")
    ids = data["customerID"].str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Training customerID values must be present and unique.")
    labels = data[TARGET].str.strip()
    if labels.isna().any() or not labels.isin(["Yes", "No"]).all():
        raise ValueError("Every Churn label must be 'Yes' or 'No'; missing labels are not allowed.")
    y = labels.map({"Yes": 1, "No": 0}).astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError("Training needs both churn classes with at least 10 rows each.")
    X = data.drop(columns=[TARGET, "customerID"])
    normalize_features(X)  # Fail early, before any expensive fitting or output creation.
    return X, y


class TelcoFeatures(TransformerMixin, BaseEstimator):
    """Persist the same input cleaning inside every trained sklearn pipeline."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TelcoFeatures:
        clean = normalize_features(X)
        empty = clean.columns[clean.isna().all()].tolist()
        if empty:
            raise ValueError(f"Cannot learn imputations for entirely missing columns: {empty}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        return normalize_features(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURES, dtype=object)
