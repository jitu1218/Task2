# Evaluation Dashboard for Model Cards Using Real Data

A complete assignment with real-data training, an interactive dashboard, documented
model cards, evaluation exports and tests.

## Open it immediately — no installation or internet needed

Extract the project ZIP and double-click **`dist/index.html`**. Keep `app.js`, `data.js`
and `style.css` beside it. No server, CDN, Python or package installation is needed
to use the dashboard. Results are already generated from trained classifiers.

Dashboard controls:
- Choose **Held-out test** or **Validation**.
- Select a model, then move its **decision threshold** slider.
- Inspect both confusion matrices and the metrics table as they update.
- ROC dots move to the current operating points; full curves and AUC stay fixed.
- **Reset policy** restores validation-selected thresholds.
- **Export metrics** downloads the currently displayed metrics as CSV.

## Reproduce training (Windows PowerShell)

Requires Python 3.11 or 3.12. Open a terminal in the extracted project directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python train.py
.venv\Scripts\python -m unittest discover -s tests -v
Start-Process .\dist\index.html
```

macOS/Linux: use `.venv/bin/python` instead. Initial package installation needs
working internet/DNS. If pip cannot resolve pypi.org, you can still open the included
dashboard offline; reconnect before installing. The dataset is bundled with
scikit-learn, so training does not download a separate dataset.

Optional front-end metric tests (requires Node.js): `node tests/dashboard.test.js`.
After retraining, reload the browser to load the regenerated data.js.

## Real dataset and imbalance

UCI Wisconsin Diagnostic Breast Cancer, loaded through `load_breast_cancer()`:
569 observations, 30 numeric features, 212 malignant and 357 benign cases. This is
**moderate imbalance** (37.3% positive), not a rare-event fraud benchmark. It is a
useful small, real dataset with interpretable false-positive/false-negative trade-offs
and reproducible offline access. We preserve all cases and recode malignant as 1.
Both models use balanced class weights on training data, and stratified splits.

## Training and selection

- Stratified training / validation / test: 341 / 114 / 114; seeds 42 and 43.
- Logistic regression: training-only StandardScaler, balanced weights, C=1.
- Random forest: 300 trees, leaf size 2, balanced weights, seed 42.
- No test fitting or resampling. Hyperparameters fixed in advance.
- Each threshold minimizes validation `5*FN + FP` over 0.05–0.95, step 0.01.
- Tie-break: higher F1, then lower threshold. Select model on validation cost/F1.
- Test outcomes are reported only after selection. Slider changes on test are
  exploratory and do not alter the documented model preference.

The illustrative error-cost assumption is educational, not a clinical recommendation.

## Results at frozen validation-selected thresholds

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.18 | 0.932 | 0.976 | 0.953 | 0.997 | 3 | 1 |
| Random forest | 0.38 | 0.976 | 0.952 | 0.964 | 0.998 | 1 | 2 |

Logistic regression was preferred on validation under the 5:1 error-cost policy.
Its test recall is higher and it misses fewer malignant cases, while the forest
produces fewer false alarms and higher precision/F1. Which trade-off matters depends
on the intended decision. See **ANALYSIS.md** for the full analysis and limitations.

## Files

- `train.py`: model fitting, threshold selection, metrics and artifact generation.
- `dist/`: offline-ready interactive HTML/CSS/JavaScript dashboard.
- `results/evaluation.json`: predictions, ROC arrays, metrics and split indices.
- `results/metrics.csv`: baseline validation/test metrics.
- `results/predictions.csv`: held-out positive-class scores and labels.
- `ANALYSIS.md`: interpretation, model cards, attribution and limitations.
- `tests/`: Python tests and cross-language metric consistency checks.
- `VALIDATION.md`: executed verification and remaining limitations.

The browser recalculates threshold-dependent metrics using stored predictions; it
does not train models or accept patient data. ROC-AUC and average precision are
computed by scikit-learn. Undefined precision/recall display as 0 (zero_division=0).

## Model card scope

For educational evaluation only, not diagnosis, screening or clinical deployment.
The historical, small dataset has no external validation here or usable demographic
attributes for subgroup fairness checks. Single-split differences are not proof of
statistical superiority. Outputs are uncalibrated scores, not validated risk estimates.

## Attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin
(Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
Dataset license: CC BY 4.0. Loaded using scikit-learn; original labels recoded.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Submission

The ZIP includes runnable source, precomputed results, analysis and the dashboard.
Upload it to Drive with viewer access or push it to GitHub, then submit that shareable
link according to your course. A private hosted dashboard link alone may not grant
your evaluator access.
