# Validation record

Review date: 27 September 2026. The 5 code cells in the included notebook were executed in order
in a fresh IPython process launched from `notebooks/`, on Linux with Python 3.12 and CPU execution.
The saved notebook contains the resulting text and figure outputs, with no saved execution errors.

Three regression tests passed: objective evaluation at updated parameters; returned weights match
the lowest recorded objective; rejected invalid labels/nonfinite inputs and prediction before fitting.
The synthetic notebook used 450 training and 150 test observations. Both implementations classified
149 test observations correctly. The scratch optimizer used its full 5,000-update budget; this is not
a certificate that it reached the global optimum.

Core package versions match the pins in `requirements.txt`. The notebook web interface and installation
on Windows/macOS were not separately exercised. Stochastic results can vary across platforms and
library builds. This validation covers the supplied example and focused regression cases, not every
possible input or production deployment.

Re-run regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```
