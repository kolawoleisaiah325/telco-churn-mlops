# Telco Customer Churn Prediction

A reproducible machine learning project for predicting whether a telecom customer leaves. It compares a baseline and three models using cross-validation, selects a decision threshold on validation data, and evaluates the selected model on a separate test set.

The project includes a training command, a prediction command, reusable Python modules, and regression tests. Python 3.12 is the verified runtime.

## Deployment lesson 1: a running API

`api.py` exposes `GET /health`. This checks that the web process can answer an HTTP request; it does not make a test prediction. `serve.py` starts the server on your computer at `127.0.0.1:8000`.

With this workspace's embedded Python, install the API packages into the project-local `.vendor` directory if they are not already present:

```powershell
Set-Location 'C:\Code\Projects\telco-customer-churn'
& 'C:\Code\Projects\python\python.exe' -m pip install --target .vendor -r requirements-api.txt
$env:CHURN_MODEL_PATH = (Resolve-Path '.\artifacts\20260923T010434Z-559283dd\churn_model.joblib').Path
& 'C:\Code\Projects\python\python.exe' .\serve.py
```

Leave that PowerShell window open. In a second PowerShell window, send one request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The response is `status: ok`. Press Ctrl+C in the first window to stop the server. The `.vendor` directory is ignored by Git. A normal Python installation should instead use a virtual environment and install both requirements files there.

## Deployment lesson 2: predict through HTTP

With the server running as above, send the JSON example from another PowerShell window:

```powershell
Set-Location 'C:\Code\Projects\telco-customer-churn'
$customer = Get-Content .\sample_customer.json -Raw
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' -ContentType 'application/json' -Body $customer
```

The API accepts one customer with the 19 features in `sample_customer.json`. It returns `customerID`, `churn_score`, `predicted_churn`, `decision_threshold`, and `model_name`. For this sample the score is about `0.754398`, the saved threshold is about `0.325706`, and the decision is `Yes`. This matches the CSV prediction command. The score is not a calibrated probability.

The request moves through four steps: FastAPI checks that the JSON has the expected fields and basic types; the saved pipeline cleans and transforms those fields; XGBoost produces a score; the saved threshold converts the score to `Yes` or `No`. Invalid JSON fields or missing required features receive HTTP 422. Visit `http://127.0.0.1:8000/docs` while the server is running to inspect the interactive API contract.

`CHURN_MODEL_PATH` selects one exact, trusted training run. The API loads that bundle once when the server starts, so every request uses the same model and threshold until the service is restarted. The generated model and source dataset are intentionally absent from GitHub. On a new machine, obtain the dataset, run training, and point `CHURN_MODEL_PATH` at the new run's `churn_model.joblib` before starting the API.

## Run it

Your dataset is already at `C:\Code\Projects\telco-customer-churn\Telco-Customer-Churn.csv`. The default command works from any directory:

```powershell
& 'C:\Code\Projects\python\python.exe' 'C:\Code\Projects\telco-customer-churn\telco_churn.py'
```

For a fresh environment, install the dependencies first. The lock file records the installed dependency graph verified on Windows and Python 3.12; the shorter `requirements.txt` pins direct dependencies.

```powershell
Set-Location 'C:\Code\Projects\telco-customer-churn'
& 'C:\Code\Projects\python\python.exe' -m pip install -r requirements-lock.txt
```

Use `--data` and `--output` for different paths, `--seed` to reproduce a split, `--folds` to change cross-validation folds, and `--jobs` to cap model threads. Defaults are seed 42, five folds, and two threads.

```powershell
& 'C:\Code\Projects\python\python.exe' .\telco_churn.py --folds 5 --jobs 2
```

A completed run is published in a new `artifacts/<run-id>/` directory. Its files appear together after saving succeeds. Old runs remain available.

| File | Purpose |
| --- | --- |
| `churn_model.joblib` | Fitted pipeline, raw-input cleaning, selected cutoff, schema version, and run metadata |
| `metrics.json` | Cross-validation scores, validation and test metrics, baseline, parameters, and environment |
| `model_comparison.csv` | Model ranking by training cross-validation average precision |
| `split_manifest.json` | Exact CSV row positions assigned to each data split |
| `exploratory_analysis.png` | Training class counts and churn rates by contract |
| `test_evaluation.png` | Selected model's test confusion matrix and precision/recall curve |

Files directly inside `artifacts/` from the earlier script are historical outputs. The new prediction command expects `churn_model.joblib` from a new run folder.

## Predict on customer rows

Replace `<run-id>` with the folder printed by training. Input needs the 19 predictor columns listed in `churn/data.py`. `customerID` is optional and is preserved when supplied. `Churn` and other extra fields are ignored. Rows retain their original order.

```powershell
Set-Location 'C:\Code\Projects\telco-customer-churn'
& 'C:\Code\Projects\python\python.exe' .\predict.py --model '.\artifacts\<run-id>\churn_model.joblib' --input '.\new_customers.csv' --output '.\predictions.csv'
```

Output contains `churn_score` and `predicted_churn`, plus `customerID` when present. The saved threshold converts scores into Yes/No decisions. Scores are uncalibrated model outputs; do not interpret a score of 0.7 as a validated 70% probability. Choose a new output filename for each prediction export; existing files are not overwritten.

Only load joblib models from a trusted source, because loading them can execute Python code. Keep the project modules available and use the recorded package versions when loading a saved model. See [scikit-learn's model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

## What happens, and why

1. **Validate the input before training.** Every required predictor must be present. Training also requires a unique, nonempty `customerID` and a Yes/No `Churn` label for every row. Repeated IDs are rejected because otherwise one customer's records could land in multiple splits. Missing or malformed targets raise an error instead of silently removing rows. Duplicate CSV headers also fail clearly.

2. **Use one cleaning process for training and predictions.** `TelcoFeatures` is the first step of the saved pipeline. It trims text, converts numeric fields, and turns blank values into missing values. The dataset has 11 blank `TotalCharges` entries. These are imputed later using training statistics. Non-numeric text, infinity, negative charges, fractional tenure, and invalid SeniorCitizen values raise errors. Entirely missing training columns also raise errors because there is no statistic to learn. IDs and target labels never enter the model.

3. **Separate the data by purpose.** The default split is 4,225 training rows, 1,409 validation rows, and 1,409 test rows. Splits are stratified to maintain approximately the same churn proportion. Training data fits and compares models; validation data chooses the cutoff; test data estimates performance after those decisions. Exploratory charts use training rows only. The saved split manifest allows you to recover the exact row membership.

4. **Keep learned preprocessing inside each cross-validation fold.** Numeric medians, category replacements, and one-hot encodings are fitted on that fold's training rows. Logistic Regression also receives numeric scaling. Tree models do not need scaling. SeniorCitizen is treated as categorical even though the source represents it with 0 and 1. Unknown prediction-time categories are handled by the encoder, though their encoded category block contains no learned category indicator. This arrangement follows [scikit-learn's guidance on avoiding preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html).

5. **Compare models using five training folds.** A prior-probability dummy model provides a basic reference. Logistic Regression, Random Forest, and XGBoost provide linear, bagged-tree, and boosted-tree alternatives. Models are ranked by mean validation-fold average precision (AP), and fold standard deviation records how much scores vary between partitions. AP summarizes the precision/recall tradeoff; it is not identical to trapezoidal PR-AUC. We use the exact name of the [metric implemented by scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Hyperparameters are fixed in `candidate_models`; this is not an exhaustive parameter search.

6. **Choose the selected model's cutoff on validation data.** The code checks every distinct validation score to maximize F1. The previous fixed search range could miss a good cutoff above 0.8 or below 0.1. Equal-F1 cutoffs choose the higher threshold, generating fewer alerts. The selected pipeline is fitted on the 60% training partition and is not refitted after threshold tuning, because refitting could change the score distribution that the cutoff was chosen for.

7. **Evaluate and preserve the actual decision rule.** The selected model's test report includes accuracy, precision, recall, F1, ROC-AUC, AP, and confusion counts. The prior baseline is evaluated at 0.5 for context. The bundle stores the fitted pipeline and threshold together. `predict.py` applies that threshold instead of falling back to the classifier's default `predict()` cutoff.

8. **Record enough information to trace a result.** Each run records a SHA-256 hash of the dataset, exact row splits, package versions, Python version, seed, model parameters, and threshold. Distinct run directories prevent a new model from being paired accidentally with an old chart or metric report. Logging identifies the current training stage, and thread limits keep the workflow practical on this computer.

## Code layout

```text
churn/
  data.py          input schema, CSV validation, and pipeline cleaning
  modeling.py      models, threshold selection, metrics, and prediction
  reporting.py     charts rendered without opening desktop windows
telco_churn.py      training command and workflow orchestration
predict.py          prediction command
tests/              regression tests
requirements.txt    pinned direct dependencies
requirements-lock.txt  verified Windows/Python 3.12 dependency graph
```

The two commands add their own project directory to the import path when needed because the Python installation in this workspace is an embedded distribution. They also work with a regular Python installation.

## Check the code

From this project directory:

```powershell
& 'C:\Code\Projects\python\python.exe' -m unittest discover -s tests -t . -v
```

Tests cover malformed and missing data, duplicate IDs and headers, single-class labels, split isolation, threshold edge cases, preprocessing stability, unseen categories, and serialization that preserves the decision threshold and row order.

## Latest verified run

Run `20260923T010434Z-559283dd` completed using all 7,043 source rows and the defaults above.

| Candidate | Mean training CV AP | Fold standard deviation |
| --- | ---: | ---: |
| XGBoost | 0.671 | 0.016 |
| Logistic Regression | 0.665 | 0.029 |
| Random Forest | 0.650 | 0.007 |
| Prior baseline | 0.265 | 0.001 |

XGBoost was selected using cross-validation. Its validation-selected cutoff was 0.325706. On the 1,409 test rows it achieved AP 0.660, ROC-AUC 0.845, precision 0.539, recall 0.749, and F1 0.627. It found 280 of the 374 churners, missed 94, and flagged 239 customers who stayed. Similar CV scores do not establish a statistically significant advantage over Logistic Regression.

Verification completed:

- All 20 regression tests passed.
- The full default training command completed and both charts were visually inspected.
- A separate invocation of `predict.py` loaded the persisted bundle and scored all 1,409 test customers from raw CSV rows. Customer order, average precision, and every confusion-matrix count matched the training report.
- An attempted repeat prediction export was rejected and the existing file remained unchanged.
- `pip check` reported no broken requirements.

The review fixed concrete issues in the earlier script: the cutoff was omitted from the saved model; cleaning was only applied during training; invalid numbers could silently become missing values; missing labels were dropped; `Contract` was used without being validated; the threshold search was artificially restricted; and the reported PR-AUC was actually AP. The revised workflow also adds cross-validation, a baseline, reproducible run records, and focused tests. Model quality should still be assessed using the limitations below.

A follow-up error check fixed three edge cases: malformed CSV records could be silently interpreted using an unintended index; empty or unreadable model files could cause uncaught loading errors; and scalar target input to the threshold helper raised a TypeError instead of a clear validation error. CSV record widths and quoting are now checked before parsing. Six additional tests cover these cases and confirm that valid quoted commas and UTF-8 BOM files remain supported. Full training and reloaded-model prediction checks passed with unchanged test metrics.

## Limits of the results

This is a tested portfolio project. F1 gives precision and recall equal importance; a real retention program needs a threshold based on its costs and capacity. Cross-validation variation is not a confidence interval. The test rows were already evaluated in the earlier version of this project, so the revised test result is not a new independent validation dataset. Repeatedly changing models based on these test results would bias the estimate. Future deployment would need new or time-ordered data and calibration checks; the current dataset has no event timestamps for a temporal evaluation.
