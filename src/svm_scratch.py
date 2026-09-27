"""Educational linear SVM using batch subgradients, not a production solver."""
import numpy as np

class LinearSVMScratch:
    """Minimize 0.5*||w||² + C*sum(hinge), with an unregularized intercept.

    Uses lr/sqrt(iteration) steps and retains the lowest objective encountered.
    Small objective changes for ten steps are a stopping heuristic, not an
    optimality certificate for this nonsmooth objective.
    """
    def __init__(self, C=1.0, lr=1e-2, max_iter=5000, tol=1e-7, random_state=42):
        if not np.isfinite(C) or C <= 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError('C and lr must be finite and positive.')
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError('max_iter must be a positive integer.')
        if not np.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative.')
        self.C, self.lr, self.max_iter, self.tol = C, lr, max_iter, tol
        self.random_state = random_state
        self.w_ = None
        self.b_ = 0.0

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a nonempty finite two-dimensional matrix.')
        return X

    def _objective(self, X, y):
        margins = y * (X @ self.w_ + self.b_)
        return float(0.5 * self.w_ @ self.w_ + self.C * np.maximum(0, 1 - margins).sum())

    def fit(self, X, y):
        X = self._matrix(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(X) or set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError('Training labels must contain both -1 and +1, one per row.')
        self.w_ = np.random.default_rng(self.random_state).normal(scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.objective_history_ = [self._objective(X, y)]
        best_value = self.objective_history_[0]
        best_w, best_b = self.w_.copy(), self.b_
        stable_steps = 0
        self.converged_ = False
        for iteration in range(1, self.max_iter + 1):
            active = y * (X @ self.w_ + self.b_) < 1
            grad_w = self.w_ - self.C * (X[active] * y[active, None]).sum(axis=0)
            grad_b = -self.C * y[active].sum()
            step = self.lr / np.sqrt(iteration)
            self.w_ -= step * grad_w
            self.b_ -= step * grad_b
            # Evaluate both terms at the UPDATED parameters.
            value = self._objective(X, y)
            if not np.isfinite(value):
                raise ValueError('Optimization diverged; scale inputs and reduce lr.')
            if value < best_value:
                best_value, best_w, best_b = value, self.w_.copy(), self.b_
            change = abs(self.objective_history_[-1] - value)
            stable_steps = stable_steps + 1 if change < self.tol else 0
            self.objective_history_.append(value)
            self.n_iter_ = iteration
            if stable_steps >= 10:
                self.converged_ = True
                break
        self.w_, self.b_ = best_w, best_b
        self.best_objective_ = best_value
        return self

    def decision_function(self, X):
        if self.w_ is None:
            raise RuntimeError('Call fit before prediction.')
        X = self._matrix(X)
        if X.shape[1] != len(self.w_):
            raise ValueError('Feature count differs from the training matrix.')
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
