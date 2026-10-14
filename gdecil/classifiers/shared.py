"""One covariance shared by all classes.

The shared covariance is estimated from the residuals (training samples minus
their class mean) of the classes seen so far, so it is recomputed after every
task. With a shared covariance the normalization term is the same for every
class and does not affect the prediction.
"""
import numpy as np
from sklearn.covariance import OAS

from .base import GaussianClassifier, class_means


class SharedCov(GaussianClassifier):
    uses_norm_term = False

    def fit(self, X, y, num_classes):
        self.means = class_means(X, y, num_classes)
        self.residuals = [X[y == c] - self.means[c] for c in range(num_classes)]

    def _pooled_residuals(self, active):
        return np.vstack([self.residuals[c] for c in active if len(self.residuals[c]) > 0])


class SharedDiagCov(SharedCov):
    """Shared diagonal covariance: variance of the pooled residuals plus 1e-6."""

    def start_task(self, active):
        self.inv_var = 1.0 / (np.var(self._pooled_residuals(active), axis=0) + 1e-6)

    def scores(self, X, active):
        return np.stack([-np.sum((X - self.means[c]) ** 2 * self.inv_var, axis=1) for c in active], axis=1)


class SharedFullCov(SharedCov):
    """Shared full covariance estimated with OAS shrinkage."""

    def start_task(self, active):
        self.precision = OAS().fit(self._pooled_residuals(active)).precision_

    def scores(self, X, active):
        out = []
        for c in active:
            diff = X - self.means[c]
            out.append(-np.sum((diff @ self.precision) * diff, axis=1))
        return np.stack(out, axis=1)
