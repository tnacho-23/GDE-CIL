"""PCA from accumulated first- and second-order statistics."""
import numpy as np


class RunningMoments:
    """Accumulates N, sum(x) and sum(x x^T) in float64, batch by batch."""

    def __init__(self, dim: int):
        self.n = 0
        self.sum = np.zeros(dim)
        self.outer_sum = np.zeros((dim, dim))

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        self.n += batch.shape[0]
        self.sum += batch.sum(axis=0)
        self.outer_sum += batch.T @ batch

    def save(self, path: str):
        np.savez(path, n=self.n, sum=self.sum, outer_sum=self.outer_sum)

    @classmethod
    def load(cls, path: str):
        with np.load(path) as data:
            moments = cls(data['sum'].shape[0])
            moments.n, moments.sum, moments.outer_sum = int(data['n']), data['sum'], data['outer_sum']
        return moments


def fit_pca(moments: RunningMoments):
    """Return (mean, components, eigenvalues) of the sample covariance
        cov = (sum(x x^T) - N mean mean^T) / (N - 1),
    with the principal axes as the columns of `components`, sorted by
    decreasing eigenvalue."""
    mean = moments.sum / moments.n
    cov = (moments.outer_sum - moments.n * np.outer(mean, mean)) / (moments.n - 1)
    eigenvalues, components = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return mean, components[:, order], eigenvalues[order]
