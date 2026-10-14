"""Mixtures of Gaussians per class, with the number of components chosen by BIC."""
import numpy as np
from sklearn.covariance import OAS
from sklearn.mixture import GaussianMixture

from .base import GaussianClassifier, gaussian_log_norm, logsumexp, mahalanobis_sq


def fit_gmm_bic(Xc, max_components, covariance_type):
    """Fit sklearn GaussianMixture models with 1..max_components components and
    return the one with the lowest BIC."""
    best_bic, best = np.inf, None
    for k in range(1, max_components + 1):
        if len(Xc) <= k:
            continue
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, n_init=1).fit(Xc)
        bic = gmm.bic(Xc)
        if bic < best_bic:
            best_bic, best = bic, gmm
    if best is None:
        raise ValueError(f"A class has only {len(Xc)} training sample(s); a mixture needs at least 2.")
    return best


def stable_inverse(cov, jitter, floor):
    """Inverse and log-determinant of cov + jitter * I, with eigenvalues floored at `floor`."""
    cov = cov.copy()
    cov.flat[::cov.shape[0] + 1] += jitter
    eigs, P = np.linalg.eigh(cov)
    eigs = np.maximum(eigs, floor)
    return P @ np.diag(1.0 / eigs) @ P.T, np.sum(np.log(eigs))


class MixtureFull(GaussianClassifier):
    """Per-class mixture of full-covariance Gaussians (MOG4-F).

    Each component covariance gets a 1e-5 jitter and eigenvalues floored at 1e-7
    before inversion.
    """

    def __init__(self, max_components: int = 4, **kw):
        super().__init__(**kw)
        self.max_components = max_components

    def fit(self, X, y, num_classes):
        dim = X.shape[1]
        self.components, self.n_components = [], []
        for c in range(num_classes):
            gmm = fit_gmm_bic(X[y == c], self.max_components, 'full')
            comps = []
            for m in range(gmm.n_components):
                precision, logdet = stable_inverse(gmm.covariances_[m], jitter=1e-5, floor=1e-7)
                comps.append((gmm.means_[m], precision, gaussian_log_norm(dim, logdet),
                              np.log(gmm.weights_[m] + 1e-12)))
            self.components.append(comps)
            self.n_components.append(gmm.n_components)

    def scores(self, X, active):
        out = []
        for c in active:
            logs = [log_w + (log_A if self.use_norm_term else 0.0) - 0.5 * mahalanobis_sq(X, mean, precision)
                    for mean, precision, log_A, log_w in self.components[c]]
            out.append(logsumexp(np.array(logs), axis=0))
        return np.stack(out, axis=1)


class MixtureShared(GaussianClassifier):
    """Per-class mixture of Gaussians with one covariance shared by all components
    of all classes (MOG4-Shared).

    1. Per class, a diagonal GaussianMixture with 1..max_components components is
       chosen by BIC; only its means, weights and the assignment of each training
       sample to a component are kept.
    2. After each task, the residuals of the active classes (samples minus the mean
       of their component) are pooled, and the shared covariance is estimated with
       OAS shrinkage, plus a 1e-5 jitter and eigenvalues floored at 1e-7.
    """
    uses_norm_term = False

    def __init__(self, max_components: int = 4, **kw):
        super().__init__(**kw)
        self.max_components = max_components

    def fit(self, X, y, num_classes):
        self.components, self.residuals, self.n_components = [], [], []
        for c in range(num_classes):
            Xc = X[y == c]
            gmm = fit_gmm_bic(Xc, self.max_components, 'diag')
            self.components.append([(gmm.means_[m], np.log(gmm.weights_[m] + 1e-12))
                                    for m in range(gmm.n_components)])
            self.residuals.append(Xc - gmm.means_[gmm.predict(Xc)])
            self.n_components.append(gmm.n_components)

    def start_task(self, active):
        pooled = np.vstack([self.residuals[c] for c in active])
        self.precision, _ = stable_inverse(OAS().fit(pooled).covariance_, jitter=1e-5, floor=1e-7)

    def scores(self, X, active):
        # The shared normalization constant is the same for every class and is omitted.
        out = []
        for c in active:
            logs = [log_w - 0.5 * mahalanobis_sq(X, mean, self.precision) for mean, log_w in self.components[c]]
            out.append(logsumexp(np.array(logs), axis=0))
        return np.stack(out, axis=1)
