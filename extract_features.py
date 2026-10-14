"""Extract frozen-backbone features for a dataset.

Writes features/{backbone}/{dataset}_raw.npz with the raw model output, or
features/{backbone}/{dataset}_l2.npz with l2-normalized features when --l2 is given (see gdecil/features.py).
Without the flag, features are not normalized.

Usage:
    python extract_features.py --backbone dinov2 --dataset cifar100
    python extract_features.py --backbone dinov2 --dataset all --l2
    python extract_features.py --backbone vitb --dataset all --l2
"""
import argparse
import os

import numpy as np

from gdecil.backbones import BACKBONES
from gdecil.datasets import BASE_DATASETS, image_dataset
from gdecil.features import add_l2_argument, feature_path, resolve_l2, save_features


def extract(backbone: str, dataset: str, l2=None, batch_size: int = 64, num_workers: int = 2,
            overwrite: bool = False):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    l2 = resolve_l2(backbone, l2)
    path = feature_path(backbone, dataset, l2)
    if os.path.exists(path) and not overwrite:
        print(f"{path} already exists (use --overwrite to extract again)")
        return path

    spec = BACKBONES[backbone]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {spec.name} on {device}")
    model, preprocess = spec.load(device)
    data = image_dataset(dataset, preprocess)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features, labels = [], []
    print(f"Extracting {len(data)} images of {dataset} ({'l2-normalized' if l2 else 'raw'} features)")
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            out = spec.forward(model, images.to(device))
            if l2:
                out = F.normalize(out, p=2, dim=1)
            features.append(out.cpu().numpy())
            labels.append(np.asarray(targets))
            print(f"  {100 * (i + 1) / len(loader):5.1f}%", end='\r')

    save_features(path, np.concatenate(features), np.concatenate(labels))
    print(f"\nSaved {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--backbone', required=True, choices=list(BACKBONES))
    p.add_argument('--dataset', required=True, choices=BASE_DATASETS + ['all'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--overwrite', action='store_true')
    add_l2_argument(p)
    args = p.parse_args()
    for dataset in (BASE_DATASETS if args.dataset == 'all' else [args.dataset]):
        extract(args.backbone, dataset, args.l2, args.batch_size, args.num_workers, args.overwrite)


if __name__ == '__main__':
    main()   # the __main__ guard is required by DataLoader workers on Windows
