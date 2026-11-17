"""Train on real bundled data; tune thresholds on validation, evaluate held-out test."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)), auc=float(roc_auc_score(y, p)),
                ap=float(average_precision_score(y, p)), tn=int(tn), fp=int(fp),
                fn=int(fn), tp=int(tp), cost=int(fp + 5 * fn))


def split_data(y):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=.4, stratify=y, random_state=42)
    val, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=43)
    return dict(train=train, validation=val, test=test)


def main():
    dataset = load_breast_cancer()
    X = dataset.data
    y = (dataset.target == 0).astype(int)  # Malignant is explicitly the positive class.
    splits = split_data(y)
    train, val, test = [splits[k] for k in ['train', 'validation', 'test']]
    estimators = {
        'lr': ('Logistic regression', make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', C=1, max_iter=3000, random_state=42))),
        'rf': ('Random forest', RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1)),
    }
    result = {'dataset': {'name': 'Wisconsin Diagnostic Breast Cancer', 'n': len(y),
        'positive': int(y.sum()), 'negative': int((y == 0).sum()), 'features': 30,
        'source': 'https://doi.org/10.24432/C5DW2B'}, 'models': [],
        'splits': {k: {'ids': v.tolist(), 'n': len(v), 'positive': int(y[v].sum()),
                       'y': y[v].tolist()} for k, v in splits.items()},
        'costs': {'false_negative': 5, 'false_positive': 1}}
    for key, (name, model) in estimators.items():
        model.fit(X[train], y[train])
        pv = model.predict_proba(X[val])[:, 1]
        grid = [(float(t), metrics(y[val], pv, t)) for t in np.round(np.arange(.05, .951, .01), 2)]
        threshold, best = min(grid, key=lambda pair: (pair[1]['cost'], -pair[1]['f1'], pair[0]))
        entry = {'id': key, 'name': name, 'threshold': threshold, 'results': {}}
        for split, ids in [('validation', val), ('test', test)]:
            p = model.predict_proba(X[ids])[:, 1]
            fpr, tpr, _ = roc_curve(y[ids], p)
            entry['results'][split] = {'p': p.tolist(), 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                'metrics': metrics(y[ids], p, threshold), 'at_half': metrics(y[ids], p, .5)}
        result['models'].append(entry)
    chosen = min(result['models'], key=lambda m: (m['results']['validation']['metrics']['cost'],
                                                  -m['results']['validation']['metrics']['f1']))
    result['selected'] = chosen['id']
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='') as file:
        fields = ['model', 'split', 'threshold', *chosen['results']['test']['metrics']]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for m in result['models']:
            for split in ['validation', 'test']:
                writer.writerow(dict(model=m['name'], split=split, threshold=m['threshold'],
                                     **m['results'][split]['metrics']))
    with (out / 'predictions.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['split', 'sample_index', 'actual_malignant', 'lr_probability', 'rf_probability'])
        for split in ['validation', 'test']:
            for j, i in enumerate(splits[split]):
                writer.writerow([split, int(i), int(y[i]), *[m['results'][split]['p'][j] for m in result['models']]])
    (ROOT / 'dist' / 'data.js').write_text('window.EVALUATION = ' + json.dumps(result) + ';', encoding='utf-8')
    analysis = ['# Model comparison and model cards', '',
      '## Data and methods',
      'Real UCI Wisconsin Diagnostic Breast Cancer data: 569 cases, 30 numeric features, '
      '212 malignant and 357 benign (37.3% / 62.7%). This is moderate natural imbalance, '
      'not rare-event fraud. Malignant is recoded as class 1. A majority-only predictor '
      'could reach 62.7% accuracy over the full dataset while finding no malignant cases.',
      '', 'All 569 cases are retained. Both estimators use class_weight="balanced", computed '
      'from training labels; stratification preserves prevalence. StandardScaler is inside '
      'the logistic pipeline and learns only from training. No resampling, test fitting, '
      'or test-based hyperparameter search occurs. The stratified split is 341 train / '
      '114 validation / 114 test. Seeds 42 and 43 are fixed.', '',
      '## Decision policy',
      'An illustrative cost of 5 per false negative and 1 per false positive is fixed before '
      'evaluation. Each threshold is selected on validation using a 0.05–0.95 grid in 0.01 '
      'steps, minimizing 5×FN+FP, with higher F1 then lower threshold breaking ties. '
      'Model selection uses validation cost, then F1. These are educational cost assumptions, '
      'not clinical recommendations. Thresholds and hyperparameters are not changed after '
      'seeing test outcomes.', '',
      f"Validation-selected model: **{chosen['name']}**, threshold **{chosen['threshold']:.2f}**.", '',
      '| Model | Threshold | Test precision | Test recall | Test F1 | ROC-AUC | FP | FN | Cost |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in result['models']:
        a = m['results']['test']['metrics']
        analysis.append(f"| {m['name']} | {m['threshold']:.2f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} | {a['auc']:.4f} | {a['fp']} | {a['fn']} | {a['cost']} |")
    analysis += ['', '## Why prefer one model?',
      f"Under the declared error-cost policy, prefer {chosen['name']} provisionally because it "
      'was selected using validation data. Compare its fixed-policy test false negatives '
      'and false positives in the table; a different deployment objective may favor a '
      'different trade-off. Logistic regression is a simpler linear baseline with inspectable '
      'coefficients (correlated features complicate interpretation). Random forest can '
      'capture nonlinear relationships but is larger and less directly interpretable. '
      'High ROC-AUC alone is not enough to select a threshold or justify deployment.', '',
      '## Interactive dashboard',
      'The split selector switches validation/test results. Selecting a model changes the '
      'threshold control. Both confusion matrices and the table update from stored real '
      'predictions. ROC curves and AUC remain invariant to threshold; markers move to the '
      'current operating points. Reset restores the original validation-selected thresholds. '
      'Test-set slider exploration is descriptive only: do not claim tuned test results '
      'as independent validation. Export downloads the currently displayed metrics.', '',
      '## Model card: intended use and limits',
      '- Intended use: educational comparison of binary classifiers on this historical benchmark.',
      '- Inputs: 30 image-derived cell-nucleus measurements; not raw images or clinical records.',
      '- Outputs: uncalibrated malignant-class scores and threshold-based predictions.',
      '- Not intended for diagnosis, screening decisions, or clinical deployment.',
      '- Small, single-source historical data; no external or prospective validation.',
      '- No demographic attributes are available here for meaningful subgroup fairness assessment.',
      '- One split has sampling uncertainty; no significance or confidence interval claims are made.',
      '- Class weighting may distort probabilities: do not interpret scores as calibrated clinical risk.',
      '- Before deployment: external validation, subgroup evaluation, calibration, governance and monitoring.', '',
      '## Attribution',
      'Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin '
      '(Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. '
      'CC BY 4.0. Loaded using scikit-learn; labels recoded to malignant=1.',
      'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html']
    (ROOT / 'ANALYSIS.md').write_text('\n\n'.join(analysis).replace('\n\n|', '\n|'), encoding='utf-8')
    print(json.dumps({'selected': chosen['name'], 'models': [dict(name=m['name'], threshold=m['threshold'],
          test=m['results']['test']['metrics']) for m in result['models']]}, indent=2))


if __name__ == '__main__':
    main()
