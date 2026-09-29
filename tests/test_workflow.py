"""Regression coverage for input quality, leakage, thresholds, and persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.data import FEATURES, TelcoFeatures, load_training_data, normalize_features, read_csv
from churn.modeling import build_pipeline, choose_threshold, make_bundle, predict_frame
from telco_churn import split_data
from predict import main as predict_main


def customers(rows: int = 60) -> pd.DataFrame:
    frame = pd.DataFrame({column: ["No"] * rows for column in FEATURES})
    frame["tenure"] = np.arange(rows)
    frame["MonthlyCharges"] = np.arange(rows) + 20.0
    frame["TotalCharges"] = frame["MonthlyCharges"] * frame["tenure"]
    frame["SeniorCitizen"] = np.arange(rows) % 2
    frame["Contract"] = "Month-to-month"
    frame["gender"] = "Female"
    frame["InternetService"] = "DSL"
    frame["PaymentMethod"] = "Electronic check"
    return frame


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "customers.csv"
        self.data = customers()
        self.data["customerID"] = [f"customer-{row}" for row in self.data.index]
        self.data["Churn"] = ["No"] * 30 + ["Yes"] * 30

    def test_blank_charges_remain_missing_for_training_imputer(self):
        self.data["TotalCharges"] = self.data["TotalCharges"].astype(object)
        self.data.loc[0, "TotalCharges"] = "   "
        self.data.to_csv(self.path, index=False)
        X, y = load_training_data(self.path)
        self.assertEqual(len(y), 60)
        self.assertTrue(pd.isna(normalize_features(X).loc[0, "TotalCharges"]))
        self.assertNotIn("customerID", X)
        self.assertNotIn("Churn", X)

    def test_missing_contract_fails_before_plotting(self):
        with self.assertRaisesRegex(ValueError, "Contract"):
            normalize_features(self.data.drop(columns="Contract"))

    def test_duplicate_customer_ids_rejected(self):
        self.data.loc[1, "customerID"] = self.data.loc[0, "customerID"]
        self.data.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            load_training_data(self.path)

    def test_missing_or_invalid_targets_are_not_silently_dropped(self):
        for label in ["", "NA", "Maybe"]:
            with self.subTest(label=label):
                self.data.loc[0, "Churn"] = label
                self.data.to_csv(self.path, index=False)
                with self.assertRaisesRegex(ValueError, "Churn label"):
                    load_training_data(self.path)

    def test_invalid_numeric_values_fail_instead_of_becoming_missing(self):
        for column, value in [("TotalCharges", "garbage"), ("MonthlyCharges", "inf"),
                              ("tenure", -1), ("tenure", 2.5), ("SeniorCitizen", 3)]:
            with self.subTest(column=column, value=value):
                frame = customers().astype(object)
                frame.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, column):
                    normalize_features(frame)

    def test_duplicate_csv_headers_rejected(self):
        self.path.write_text("Churn,Churn\nYes,No\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_csv(self.path)

    def test_csv_record_width_must_match_header(self):
        for contents in ["a,b\nextra,value1,value2\n", "a,b\nvalue1\n"]:
            with self.subTest(contents=contents):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "expected 2"):
                    read_csv(self.path)

    def test_invalid_csv_quotes_rejected(self):
        self.path.write_text('a,b\n"unfinished,value\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed CSV"):
            read_csv(self.path)

    def test_csv_bom_and_quoted_commas_are_supported(self):
        self.path.write_text('customerID,Contract\n001,"Plan, annual"\n', encoding="utf-8-sig")
        parsed = read_csv(self.path)
        self.assertEqual(parsed.loc[0, "customerID"], "001")
        self.assertEqual(parsed.loc[0, "Contract"], "Plan, annual")

    def test_prediction_handles_empty_model_without_creating_output(self):
        model_path = self.path.with_suffix(".joblib")
        model_path.touch()
        output = self.path.parent / "predictions.csv"
        with self.assertLogs(level="ERROR") as logs:
            code = predict_main([
                "--model", str(model_path), "--input", str(self.path), "--output", str(output),
            ])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        self.assertIn("empty or truncated", logs.output[0])

    def test_prediction_handles_unreadable_pickle_without_traceback(self):
        import pickle

        with patch("predict.joblib.load", side_effect=pickle.UnpicklingError("Invalid model")):
            with self.assertLogs(level="ERROR"):
                code = predict_main([
                    "--model", "invalid.joblib", "--input", str(self.path),
                    "--output", str(self.path.parent / "predictions.csv"),
                ])
        self.assertEqual(code, 1)

    def test_single_class_rejected(self):
        self.data["Churn"] = "No"
        self.data.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "both churn classes"):
            load_training_data(self.path)

    def test_completely_missing_training_feature_rejected(self):
        frame = customers()
        frame["TotalCharges"] = np.nan
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            TelcoFeatures().fit(frame)


class ModelingTests(unittest.TestCase):
    def test_threshold_can_exceed_old_grid_limit(self):
        threshold, f1 = choose_threshold(np.array([0, 0, 1, 1]), np.array([.1, .9, .95, .99]))
        self.assertEqual(threshold, .95)
        self.assertEqual(f1, 1.0)

    def test_threshold_ties_choose_fewer_alerts(self):
        threshold, _ = choose_threshold(np.array([1, 0, 0, 1]), np.array([.9, .8, .7, .6]))
        self.assertEqual(threshold, .9)

    def test_invalid_probability_input_rejected(self):
        for scores in [np.array([np.nan, .5]), np.array([-.1, .5]), np.array([.5])]:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    choose_threshold(np.array([0, 1]), scores)

    def test_scalar_or_multidimensional_target_rejected_clearly(self):
        for labels in [np.array(1), np.array([[0], [1]])]:
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "aligned"):
                    choose_threshold(labels, np.array([.2, .8]))

    def test_split_is_disjoint_complete_and_repeatable(self):
        X = customers()
        y = pd.Series([0, 1] * 30)
        first = split_data(X, y, 42)
        second = split_data(X, y, 42)
        train, valid, test = [set(part.index) for part in first[:3]]
        self.assertFalse(train & valid or train & test or valid & test)
        self.assertEqual(train | valid | test, set(X.index))
        for a, b in zip(first, second):
            self.assertTrue(a.equals(b))

    def test_prediction_does_not_relearn_imputation_or_scaling(self):
        X = customers()
        y = pd.Series([0] * 30 + [1] * 30)
        pipeline = build_pipeline(LogisticRegression(max_iter=1000), scale_numeric=True).fit(X, y)
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        medians = numeric.named_steps["impute"].statistics_.copy()
        means = numeric.named_steps["scale"].mean_.copy()
        future = customers(2)
        future["MonthlyCharges"] = [1_000_000, np.nan]
        future["Contract"] = "Future contract type"
        scores = pipeline.predict_proba(future)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, medians)
        np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, means)

    def test_saved_bundle_preserves_cleaning_threshold_and_row_order(self):
        X = customers()
        y = pd.Series([0] * 30 + [1] * 30)
        pipeline = build_pipeline(LogisticRegression(max_iter=1000), scale_numeric=True).fit(X, y)
        raw = X.iloc[[8, 2, 51]].astype("string")
        raw["customerID"] = ["008", "002", "051"]
        raw["Churn"] = "ignored"
        raw.loc[8, "TotalCharges"] = " "
        raw.loc[2, "Contract"] = "New plan"
        # An explicit non-default cutoff must survive serialization and drive labels.
        bundle = make_bundle(pipeline, 0.0, "Test model", {})
        before = predict_frame(bundle, raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
            after = predict_frame(restored, raw)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(after["customerID"].tolist(), ["008", "002", "051"])
        self.assertEqual(after["predicted_churn"].tolist(), ["Yes", "Yes", "Yes"])
        self.assertIn(0, pipeline.predict(raw))


if __name__ == "__main__":
    unittest.main()
