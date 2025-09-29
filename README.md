# Linear SVM from Scratch (Portfolio Sample)

This repository implements a **soft-margin linear SVM** from scratch using NumPy,
trained by batch gradient descent on the hinge loss with L2 regularization.
It also includes a brief **Gaussian kernel Gram matrix** demo (educational) and
clean plots of the learned decision boundary on synthetic data.

## What this shows
- From-scratch optimization of **hinge loss + L2** (soft-margin SVM)
- Clear, reproducible notebook with decision boundary visualizations
- Educational **Gram matrix** (RBF) construction
- Comparison to `scikit-learn`'s `LinearSVC` as a sanity check

## Structure
```
.
├── notebooks
│   └── svm_from_scratch.ipynb   # Training, plots, Gram matrix demo
├── src
│   └── svm_scratch.py           # Minimal SVM class (NumPy)
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/svm_from_scratch.ipynb
```

## Notes
- The scratch model is **linear**; the RBF Gram matrix cell is for intuition/visuals, not a full kernel-SVM solver.
- Plots are made with `matplotlib` only.
- Code is kept concise and readable for portfolio review.

## License
MIT
