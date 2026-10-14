"""Class-incremental evaluation protocol and metrics."""
import time

import numpy as np

from .datasets import DATASETS
from .features import task_groups


def evaluate(classifier, X_test, y_test, dataset: str, run: int, log=print) -> np.ndarray:
    """Evaluate a fitted classifier on one task order.

    Returns R (n_tasks x n_tasks), where R[k, j] is the accuracy on the test
    samples of task j after learning tasks 0..k. Prediction is the arg-max of the
    class log-likelihoods over the classes seen so far (uniform class prior).
    """
    groups = task_groups(dataset, run)
    n_tasks = len(groups)
    R = np.zeros((n_tasks, n_tasks))
    t0 = time.time()
    for k in range(n_tasks):
        active = np.concatenate(groups[:k + 1])
        classifier.start_task(active)
        for j in range(k + 1):
            mask = np.isin(y_test, groups[j])
            if not mask.any():
                continue
            preds = active[np.argmax(classifier.scores(X_test[mask], active), axis=1)]
            R[k, j] = np.mean(preds == y_test[mask])
        log(f"  task {k + 1:2d}/{n_tasks} | average accuracy {100 * R[k, :k + 1].mean():6.2f}% | {time.time() - t0:6.1f}s")
    return R


def average_accuracy(R: np.ndarray) -> float:
    """A_T: mean accuracy over all tasks after the last one."""
    return float(R[-1].mean())


def forgetting(R: np.ndarray) -> float:
    """F_T: mean over the first T-1 tasks of (best earlier accuracy - final accuracy)."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)]))


def run_protocol(classifier, split, dataset: str, runs: int = 1, log=print):
    """Fit `classifier` on the training split once and evaluate `runs` task orders.
    Returns the list of accuracy matrices."""
    X_train, y_train, X_test, y_test = split
    t0 = time.time()
    classifier.fit(X_train, y_train, DATASETS[dataset].num_classes)
    log(f"fitted in {time.time() - t0:.1f}s")
    results = []
    for run in range(runs):
        log(f"run {run + 1}/{runs}")
        results.append(evaluate(classifier, X_test, y_test, dataset, run, log))
    return results
