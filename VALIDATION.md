# Verification record

- Training completed on all 569 real dataset observations using disjoint splits.
- Six Python tests passed: counts, split isolation, known confusion counts, zero
  positive predictions, reproduced artifact metrics and validation-only selection.
- 41 JavaScript checks passed against Python-generated baseline metrics and extremes.
- Browser QA confirmed rendering of charts, tables and model cards.
- Threshold changed to 1: logistic recall became 0, all 42 test positives were missed,
  precision displayed 0, and ROC-AUC stayed 0.997 as expected.
- Reset, model selector, validation/test switching and CSV download were exercised.
- Desktop layout visually inspected. Responsive CSS is included; mobile was not
  separately tested in a device emulator. Python training tested on Linux; Windows
  instructions supplied but not executed on Windows.

The included dashboard runs offline with precomputed predictions. Python training
requires the packages in requirements.txt. No clinical validation is claimed.
