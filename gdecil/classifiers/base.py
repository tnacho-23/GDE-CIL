"""Common interface of the class-incremental Gaussian classifiers.

The protocol (gdecil.protocol) uses a classifier in three steps:
    fit(X_train, y_train, num_classes)   once; per-class statistics, which do not
                                          depend on the order of the tasks
    start_task(active)                   after each task; shared statistics that
                                          depend on the set of classes seen so far
    scores(X, active)                    (N, len(active)) class log-likelihoods
"""
import numpy as np

LOG2PI = np.log(2 * np.pi)


class GaussianClassifier:
    uses_norm_term = True   # False when every class shares the same normalization term

    def __init__(self, use_norm_term: bool = True):
        # If False, the Gaussian normalization term (log-determinant and log(2*pi)
        # constant) is dropped and classes are ranked by Mahalanobis distance only.
        # Irrelevant for covariances shared by all classes.
        self.use_norm_term = use_norm_term

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        raise NotImplementedError

    def start_task(self, active: np.ndarray):
        pass

    def scores(self, X: np.ndarray, active: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _log_norm(self, log_A: np.ndarray) -> np.ndarray:
        return log_A if self.use_norm_term else np.zeros_like(log_A)


def gaussian_log_norm(dim: int, logdet_cov) -> np.ndarray:
    """log of the Gaussian normalization constant: -(dim log(2 pi) + log|Sigma|) / 2."""
    return -0.5 * (dim * LOG2PI + logdet_cov)


def mahalanobis_sq(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every row of X to `mean` (precision = inverse covariance)."""
    diff = X - mean
    return np.sum((diff @ precision) * diff, axis=1)


def logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis)
    return m + np.log(np.sum(np.exp(a - np.expand_dims(m, axis)), axis=axis))


def class_means(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    means = np.zeros((num_classes, X.shape[1]))
    for c in range(num_classes):
        Xc = X[y == c]
        if len(Xc) > 0:
            means[c] = Xc.mean(axis=0)
    return means
