# Churn model card

This is a demonstration model trained on IBM's fictional Telco Customer Churn sample. It is included so the API container can start from a fresh checkout.

- Training run: `20260923T010434Z-559283dd`
- Dataset: 7,043 rows; SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- Selected pipeline: XGBoost with training-fitted preprocessing
- Decision threshold: `0.3257061839103699`, chosen on validation data to maximize F1
- Held-out test: average precision `0.660`, ROC-AUC `0.845`, precision `0.539`, recall `0.749`, F1 `0.627`
- Artifact SHA-256: `89f796da725025cb714ba3930acfe4ef3a72a3c2382ff6dddaaa3a09fd02b2a8`
- Training runtime: Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1

Scores have not been calibrated as probabilities. The dataset lacks event timestamps and does not support a temporal performance check. The test set has been used during development, so its metrics should be treated as a portfolio demonstration rather than an unbiased production estimate. No real customer decisions should rely on this model without fresh validation and a business-specific threshold.

The artifact uses joblib serialization, which can execute code when loaded. Treat it as trusted project code and verify its hash before using it outside this repository.
