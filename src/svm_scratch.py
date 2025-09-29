import numpy as np

class LinearSVMScratch:
    """Soft-margin linear SVM trained with batch gradient descent on hinge loss.
    Objective: 0.5 * ||w||^2 + C * sum(max(0, 1 - y*(w^T x + b)))
    y in {-1, +1}
    """
    def __init__(self, C=1.0, lr=1e-2, max_iter=1000, tol=1e-5, random_state=42):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.w_ = rng.normal(scale=0.01, size=d)
        self.b_ = 0.0

        prev_obj = np.inf
        for _ in range(self.max_iter):
            margins = y * (X @ self.w_ + self.b_)
            # hinge mask: 1 if margin < 1 else 0
            mask = margins < 1.0

            # gradients
            grad_w = self.w_.copy() - self.C * (X[mask] * y[mask, None]).sum(axis=0)
            grad_b = - self.C * y[mask].sum()

            # update
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

            # objective
            obj = 0.5 * np.dot(self.w_, self.w_) + self.C * np.maximum(0.0, 1.0 - margins).sum()

            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj
        return self

    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
