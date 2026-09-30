# Linear SVM from Scratch

[![Checks](https://github.com/KianaAbrisham/svm-from-scratch/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/svm-from-scratch/actions/workflows/checks.yml)

An educational NumPy implementation of a soft-margin linear support vector machine,
with an executed synthetic-data example and a scikit-learn comparison.

The objective is `0.5 * ||w||² + C * sum(max(0, 1 - y * (Xw + b)))`, with labels
`-1` and `+1` and an unregularized intercept. Batch **subgradient** steps decrease as
`lr / sqrt(iteration)`. The model retains the parameters with the lowest observed objective.
The stopping rule based on small objective changes is a heuristic, not an optimality certificate.

## Example and comparison

- Generate 600 synthetic observations with two features.
- Reserve 150 test observations; fit the scaler on 450 training observations only.
- Compare with `LinearSVC(loss='hinge')` using fixed parameters.
- Plot the training objective, decision boundary and held-out observations.
- Build an RBF Gram matrix separately to illustrate a kernel similarity matrix.

The reference uses a different optimizer and also regularizes its synthetic intercept feature.
Identical coefficients are not expected. This is a linear classifier; the RBF matrix is not a
kernel-SVM implementation. The optimizer is sensitive to scale and learning rate and is not
intended to replace a production solver.

## Files

| Path | Purpose |
|---|---|
| [notebooks/svm_from_scratch.ipynb](notebooks/svm_from_scratch.ipynb) | Executed training and visualizations |
| [src/svm_scratch.py](src/svm_scratch.py) | Validated inputs, optimization and prediction |
| [tests/test_svm.py](tests/test_svm.py) | Updated-parameter objective and retained-model checks |

## Run locally

Use Python 3.12 and a separate environment for this project. From the repository folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook notebooks/svm_from_scratch.ipynb
```

The notebook finds the repository from either its root folder or `notebooks/`.
The saved outputs come from CPU execution with the included data; see
[validation](docs/VALIDATION.md) for the checks and limits.

[Development notes](https://github.com/KianaAbrisham/KianaAbrisham/blob/main/docs/DEVELOPMENT.md)

## License

MIT — see [LICENSE](LICENSE).
