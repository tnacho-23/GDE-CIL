"""Feature files, train/test split, PCA reduction and task construction.

Feature file: features/{backbone}/{dataset}_{l2|raw}.npz with
    features  float32 (N, dim), l2-normalized ('l2') or as output by the model ('raw')
    labels    int64 (N,), sorted so that each class is a contiguous block

Whether features are l2-normalized is chosen with --l2 / --no-l2 in every script;
by default it follows the backbone (Backbone.l2_normalize, off for all backbones).
"""
import argparse
import os

import numpy as np

from .backbones import BACKBONES
from .datasets import DATASETS, JOINT_COMPONENTS
from .paths import FEATURES_DIR

SPLIT_SEED = 42
TRAIN_FRACTION = 0.8


def add_l2_argument(parser: argparse.ArgumentParser):
    parser.add_argument('--l2', action=argparse.BooleanOptionalAction, default=None,
                        help="l2-normalize the features (--l2) or keep them raw (--no-l2); "
                             "default: raw")


def resolve_l2(backbone: str, l2=None) -> bool:
    """`l2` if given, otherwise the backbone default."""
    return BACKBONES[backbone].l2_normalize if l2 is None else l2


def _norm_tag(l2: bool) -> str:
    return 'l2' if l2 else 'raw'


def feature_path(backbone: str, dataset: str, l2: bool, root: str = FEATURES_DIR) -> str:
    return os.path.join(root, backbone, f'{dataset}_{_norm_tag(l2)}.npz')


def pca_path(backbone: str, l2: bool, root: str = FEATURES_DIR) -> str:
    return os.path.join(root, backbone, f'pca_imagenet21k_{_norm_tag(l2)}.npz')


def moments_path(backbone: str, l2: bool, root: str = FEATURES_DIR) -> str:
    return os.path.join(root, backbone, f'imagenet21k_moments_{_norm_tag(l2)}.npz')


def save_features(path: str, features: np.ndarray, labels: np.ndarray):
    """Save features with labels, stably sorted by label."""
    order = np.argsort(labels, kind='stable')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, features=np.asarray(features, dtype=np.float32)[order],
             labels=np.asarray(labels, dtype=np.int64)[order])


def load_features(backbone: str, dataset: str, l2: bool, root: str = FEATURES_DIR):
    path = feature_path(backbone, dataset, l2, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python extract_features.py --backbone {backbone} "
            f"--dataset {dataset} {'--l2' if l2 else '--no-l2'}")
    with np.load(path) as data:
        return data['features'], data['labels']


def _split_balanced(features, labels, num_classes, rng, label_offset=0):
    """Per-class 80/20 split of classes 0..num_classes-1.

    The training part of every class is then truncated to the size of the
    smallest one, so all classes have the same number of training samples (per-
    class full covariances estimated from different sample counts have different
    ranks and are not comparable). The test part is kept whole.
    """
    train_parts, test_parts = [], []
    for c in range(num_classes):
        fc = features[labels == c]
        perm = rng.permutation(len(fc))
        cut = int(TRAIN_FRACTION * len(fc))
        train_parts.append(fc[perm[:cut]])
        test_parts.append(fc[perm[cut:]])

    n_train = min(len(p) for p in train_parts)
    X_train = np.concatenate([p[:n_train] for p in train_parts])
    y_train = np.repeat(np.arange(num_classes) + label_offset, n_train)
    X_test = np.concatenate(test_parts)
    y_test = np.concatenate([np.full(len(p), c + label_offset) for c, p in enumerate(test_parts)])
    return X_train, y_train, X_test, y_test


def load_split(backbone: str, dataset: str, l2: bool, root: str = FEATURES_DIR):
    """Train/test split of `dataset`: returns (X_train, y_train, X_test, y_test).

    For 'joint', each of the three datasets is split and balanced on its own (a
    global minimum would cut every class to the ~40 samples of the smallest
    ImageNet-R class) and labels are shifted by the offsets in JOINT_COMPONENTS.
    """
    rng = np.random.RandomState(SPLIT_SEED)
    components = JOINT_COMPONENTS if dataset == 'joint' else [(dataset, 0)]
    parts = []
    for name, offset in components:
        features, labels = load_features(backbone, name, l2, root)
        parts.append(_split_balanced(features, labels, DATASETS[name].num_classes, rng, offset))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def load_pca(backbone: str, l2: bool, root: str = FEATURES_DIR):
    """Return (mean, components) of the PCA fitted by fit_pca.py on features with the
    same normalization; components has the principal axes as columns, sorted by
    decreasing variance."""
    path = pca_path(backbone, l2, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python fit_pca.py --backbone {backbone} {'--l2' if l2 else '--no-l2'}")
    with np.load(path) as data:
        return data['mean'], data['components']


def project(X: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int) -> np.ndarray:
    """Center X and keep its coordinates on the first `dim` principal axes."""
    if not 1 <= dim <= components.shape[1]:
        raise ValueError(f"PCA dimension must be between 1 and {components.shape[1]}, got {dim}")
    return ((X - mean) @ components[:, :dim]).astype(X.dtype)


def task_groups(dataset: str, run: int):
    """Classes of each task for run `run` (seed 42 + run).

    Base datasets: classes are shuffled and cut into tasks of classes_per_task.
    'joint': the 550 labels are cut into 30 consecutive blocks, presented in a
    random order.
    """
    spec = DATASETS[dataset]
    rng = np.random.RandomState(SPLIT_SEED + run)
    if spec.classes_per_task is None:
        blocks = np.array_split(np.arange(spec.num_classes), spec.n_tasks)
        return [blocks[i] for i in rng.permutation(spec.n_tasks)]
    order = rng.permutation(spec.num_classes)
    cpt = spec.classes_per_task
    return [order[t * cpt:(t + 1) * cpt] for t in range(spec.n_tasks)]
