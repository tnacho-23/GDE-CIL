"""Datasets: class-incremental protocol and image sources.

Every base dataset is used with a per-class 80/20 train/test split (see
features.load_split). 'joint' concatenates the three base datasets (550 classes)
into 30 tasks presented in a random order.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .paths import DATASETS_DIR


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int        # classes used in the tasks (the first num_classes labels)
    n_tasks: int
    classes_per_task: Optional[int] = None


DATASETS = {
    'cifar100':   DatasetSpec(num_classes=100, n_tasks=10, classes_per_task=10),
    'imagenetr':  DatasetSpec(num_classes=200, n_tasks=10, classes_per_task=20),
    'caltech256': DatasetSpec(num_classes=250, n_tasks=10, classes_per_task=25),
    'joint':      DatasetSpec(num_classes=550, n_tasks=30),
}

BASE_DATASETS = ['cifar100', 'imagenetr', 'caltech256']

# Composition of 'joint': (dataset, label offset), in this order.
JOINT_COMPONENTS = [('cifar100', 0), ('caltech256', 100), ('imagenetr', 350)]


def _to_rgb(img):
    """Some Caltech-256 images are grayscale or RGBA. Module-level so DataLoader
    workers can pickle it."""
    return img.convert('RGB')


class ImageFolderDataset:
    """ImageNet-R layout: one sub-folder per class. Uses the first `num_classes`
    folders and the files of each folder in sorted order. A plain map-style
    dataset (torch's DataLoader only needs __len__ and __getitem__)."""

    EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

    def __init__(self, root: str, transform=None, num_classes: Optional[int] = None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.classes = classes[:num_classes] if num_classes is not None else classes
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        for label, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            for fname in sorted(os.listdir(folder)):
                path = os.path.join(folder, fname)
                if os.path.isfile(path) and fname.lower().rsplit('.', 1)[-1] in self.EXTENSIONS:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return (self.transform(img) if self.transform else img), label


def image_dataset(dataset: str, preprocess, root: str = DATASETS_DIR):
    """Dataset with the images of `dataset`, transformed by `preprocess`.

    cifar100:   the 50,000 training images of torchvision CIFAR100
    caltech256: all images of torchvision Caltech256 (257 classes incl. clutter)
    imagenetr:  the first 200 classes of <root>/imagenet-r
    """
    import torchvision
    import torchvision.transforms as T

    if dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=preprocess)
    if dataset == 'caltech256':
        return torchvision.datasets.Caltech256(root=root, download=False,
                                               transform=T.Compose([T.Lambda(_to_rgb), preprocess]))
    if dataset == 'imagenetr':
        return ImageFolderDataset(os.path.join(root, 'imagenet-r'), preprocess, num_classes=200)
    raise ValueError(f"No image source for dataset '{dataset}'")

