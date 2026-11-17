# Model comparison and model cards



## Data and methods

Real UCI Wisconsin Diagnostic Breast Cancer data: 569 cases, 30 numeric features, 212 malignant and 357 benign (37.3% / 62.7%). This is moderate natural imbalance, not rare-event fraud. Malignant is recoded as class 1. A majority-only predictor could reach 62.7% accuracy over the full dataset while finding no malignant cases.



All 569 cases are retained. Both estimators use class_weight="balanced", computed from training labels; stratification preserves prevalence. StandardScaler is inside the logistic pipeline and learns only from training. No resampling, test fitting, or test-based hyperparameter search occurs. The stratified split is 341 train / 114 validation / 114 test. Seeds 42 and 43 are fixed.



## Decision policy

An illustrative cost of 5 per false negative and 1 per false positive is fixed before evaluation. Each threshold is selected on validation using a 0.05–0.95 grid in 0.01 steps, minimizing 5×FN+FP, with higher F1 then lower threshold breaking ties. Model selection uses validation cost, then F1. These are educational cost assumptions, not clinical recommendations. Thresholds and hyperparameters are not changed after seeing test outcomes.



Validation-selected model: **Logistic regression**, threshold **0.18**.


| Model | Threshold | Test precision | Test recall | Test F1 | ROC-AUC | FP | FN | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.18 | 0.9318 | 0.9762 | 0.9535 | 0.9970 | 3 | 1 | 8 |
| Random forest | 0.38 | 0.9756 | 0.9524 | 0.9639 | 0.9977 | 1 | 2 | 11 |



## Why prefer one model?

Under the declared error-cost policy, prefer Logistic regression provisionally because it was selected using validation data. Compare its fixed-policy test false negatives and false positives in the table; a different deployment objective may favor a different trade-off. Logistic regression is a simpler linear baseline with inspectable coefficients (correlated features complicate interpretation). Random forest can capture nonlinear relationships but is larger and less directly interpretable. High ROC-AUC alone is not enough to select a threshold or justify deployment.



## Interactive dashboard

The split selector switches validation/test results. Selecting a model changes the threshold control. Both confusion matrices and the table update from stored real predictions. ROC curves and AUC remain invariant to threshold; markers move to the current operating points. Reset restores the original validation-selected thresholds. Test-set slider exploration is descriptive only: do not claim tuned test results as independent validation. Export downloads the currently displayed metrics.



## Model card: intended use and limits

- Intended use: educational comparison of binary classifiers on this historical benchmark.

- Inputs: 30 image-derived cell-nucleus measurements; not raw images or clinical records.

- Outputs: uncalibrated malignant-class scores and threshold-based predictions.

- Not intended for diagnosis, screening decisions, or clinical deployment.

- Small, single-source historical data; no external or prospective validation.

- No demographic attributes are available here for meaningful subgroup fairness assessment.

- One split has sampling uncertainty; no significance or confidence interval claims are made.

- Class weighting may distort probabilities: do not interpret scores as calibrated clinical risk.

- Before deployment: external validation, subgroup evaluation, calibration, governance and monitoring.



## Attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. CC BY 4.0. Loaded using scikit-learn; labels recoded to malignant=1.

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html