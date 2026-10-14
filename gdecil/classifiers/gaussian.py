"""One Gaussian per class: diagonal, full, OAS and Ledoit-Wolf covariances."""
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import OAS

from .base import GaussianClassifier, gaussian_log_norm, mahalanobis_sq


class PerClassGaussian(GaussianClassifier):
    """Classes with a mean, a precision matrix and a log-normalization constant."""

    def fit(self, X, y, num_classes):
        dim = X.shape[1]
        self.means = np.zeros((num_classes, dim))
        self.precisions = np.zeros((num_classes, dim, dim))
        self.log_A = np.zeros(num_classes)
        for c in range(num_classes):
            Xc = X[y == c]
            if len(Xc) > 1:
                self.means[c], self.precisions[c], self.log_A[c] = self._fit_class(Xc)
            else:
                self.precisions[c] = np.eye(dim)
                self.log_A[c] = gaussian_log_norm(dim, 0.0)

    def _fit_class(self, Xc):
        """Return (mean, precision, log normalization constant) of one class."""
        raise NotImplementedError

    def scores(self, X, active):
        log_A = self._log_norm(self.log_A[active])
        return np.stack([log_A[i] - 0.5 * mahalanobis_sq(X, self.means[c], self.precisions[c])
                         for i, c in enumerate(active)], axis=1)


class DiagCov(GaussianClassifier):
    """Diagonal covariance: per-dimension variances plus eps."""

    def __init__(self, eps: float = 1e-6, **kw):
        super().__init__(**kw)
        self.eps = eps

    def fit(self, X, y, num_classes):
        dim = X.shape[1]
        self.means = np.zeros((num_classes, dim))
        self.variances = np.ones((num_classes, dim))
        for c in range(num_classes):
            Xc = X[y == c]
            if len(Xc) > 1:
                self.means[c] = Xc.mean(axis=0)
                self.variances[c] = Xc.var(axis=0, ddof=1) + self.eps
        self.log_A = gaussian_log_norm(dim, np.sum(np.log(self.variances), axis=1))

    def scores(self, X, active):
        log_A = self._log_norm(self.log_A[active])
        return np.stack([log_A[i] - 0.5 * np.sum((X - self.means[c]) ** 2 / self.variances[c], axis=1)
                         for i, c in enumerate(active)], axis=1)


class FullCov(GaussianClassifier):
    """Full sample covariance plus eps * I. The precision is computed from a
    Cholesky factor (Sigma = L L^T, precision = L^-T L^-1), as in sklearn's
    GaussianMixture, instead of a direct inversion."""

    def __init__(self, eps: float = 1e-6, **kw):
        super().__init__(**kw)
        self.eps = eps

    def fit(self, X, y, num_classes):
        dim = X.shape[1]
        self.means = np.zeros((num_classes, dim))
        self.precisions_chol = np.zeros((num_classes, dim, dim))
        self.log_A = np.zeros(num_classes)
        for c in range(num_classes):
            Xc = X[y == c]
            if len(Xc) > 1:
                self.means[c] = Xc.mean(axis=0)
                cov = np.cov(Xc, rowvar=False)
                cov.flat[::dim + 1] += self.eps
                chol = np.linalg.cholesky(cov)
                self.precisions_chol[c] = solve_triangular(chol, np.eye(dim), lower=True).T
                # log|Sigma| = -2 sum(log diag(precision Cholesky factor))
                self.log_A[c] = gaussian_log_norm(dim, -2 * np.sum(np.log(np.diag(self.precisions_chol[c]))))
            else:
                self.precisions_chol[c] = np.eye(dim)
                self.log_A[c] = gaussian_log_norm(dim, 0.0)

    def scores(self, X, active):
        log_A = self._log_norm(self.log_A[active])
        out = []
        for i, c in enumerate(active):
            z = (X - self.means[c]) @ self.precisions_chol[c]
            out.append(log_A[i] - 0.5 * np.sum(z * z, axis=1))
        return np.stack(out, axis=1)


class OASCov(PerClassGaussian):
    """Oracle Approximating Shrinkage covariance (sklearn.covariance.OAS)."""

    def _fit_class(self, Xc):
        oas = OAS().fit(Xc)
        _, logdet = np.linalg.slogdet(oas.covariance_)
        return Xc.mean(axis=0), oas.precision_, gaussian_log_norm(Xc.shape[1], logdet)


class LedoitWolfCov(PerClassGaussian):
    """Ledoit-Wolf shrinkage of the sample covariance S towards a target T:
        Sigma = (1 - g) S + g T,   g = clip(beta^2 / ||S - T||_F^2, 0, 1),
        beta^2 = sum_i ||x_i x_i^T - S||_F^2 / n^2   (x_i centered samples).
    target='identity': T = mean(diag(S)) I.   target='diag': T = diag(S).
    The precision is inv(Sigma + 1e-6 I).
    """

    def __init__(self, target: str = 'identity', **kw):
        super().__init__(**kw)
        if target not in ('identity', 'diag'):
            raise ValueError(f"Unknown Ledoit-Wolf target: {target}")
        self.target = target

    def _fit_class(self, Xc):
        n, dim = Xc.shape
        mean = Xc.mean(axis=0)
        Z = Xc - mean
        S = Z.T @ Z / (n - 1)
        T = np.trace(S) / dim * np.eye(dim) if self.target == 'identity' else np.diag(np.diag(S))
        delta_sq = np.linalg.norm(S - T, 'fro') ** 2
        # sum_i ||z_i z_i^T - S||_F^2 = sum_i ||z_i||^4 - 2 sum_i z_i^T S z_i + n ||S||_F^2
        sq_norms = np.sum(Z * Z, axis=1)
        beta_sq = (np.sum(sq_norms ** 2) - 2 * np.sum((Z @ S) * Z) + n * np.sum(S * S)) / n ** 2
        gamma = min(max(beta_sq / delta_sq, 0.0), 1.0) if delta_sq != 0 else 0.0
        sigma = (1 - gamma) * S + gamma * T
        precision = np.linalg.inv(sigma + 1e-6 * np.eye(dim))
        _, logdet = np.linalg.slogdet(sigma)
        return mean, precision, gaussian_log_norm(dim, logdet)
