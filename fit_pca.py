"""Fit a PCA projection on ImageNet-21K-P features of a backbone.

ImageNet-21K-P is read from datasets/imagenet21k/ (or --data-root), which must
contain one or more folders with one sub-folder of images per class, e.g.

    datasets/imagenet21k/
        imagenet21k_train/n00005787/*.JPEG
        imagenet21k_val/n00005787/*.JPEG

Every image is used and labels are ignored. Features are l2-normalized or not
with the same --l2 / --no-l2 flag as extract_features.py (default: raw features).
Two files are written, with suffix _l2 or _raw:

    features/{backbone}/imagenet21k_moments_{l2|raw}.npz   N, sum(x), sum(x x^T)
    features/{backbone}/pca_imagenet21k_{l2|raw}.npz       mean, components, eigenvalues

`components` holds the principal axes as columns, by decreasing variance; it is
what train.py --pca-dim uses on features with the same normalization.

Usage:
    python fit_pca.py --backbone dinov2
    python fit_pca.py --backbone dinov2 --data-root /path/to/imagenet21k
    python fit_pca.py --backbone dinov2 --l2
    python fit_pca.py --backbone dinov2 --moments features/dinov2/imagenet21k_moments_raw.npz

--moments skips the extraction and refits the PCA from moments saved by a
previous run.
"""
import argparse
import os

import numpy as np

from gdecil.backbones import BACKBONES
from gdecil.features import add_l2_argument, moments_path, pca_path, resolve_l2
from gdecil.paths import DATASETS_DIR
from gdecil.pca import RunningMoments, fit_pca


def _safe_loader(path):
    """Load an image, replacing unreadable files with a black image."""
    from PIL import Image
    from torchvision.datasets.folder import pil_loader
    try:
        return pil_loader(path)
    except (OSError, Image.DecompressionBombError) as e:
        print(f"\nSkipping unreadable image {path}: {e}")
        return Image.new('RGB', (224, 224))


def accumulate_moments(backbone: str, data_root: str, l2: bool, batch_size: int, num_workers: int) -> RunningMoments:
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch.utils.data import ConcatDataset, DataLoader

    spec = BACKBONES[backbone]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = spec.load(device)
    folders = sorted(os.path.join(data_root, d) for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)))
    data = ConcatDataset([torchvision.datasets.ImageFolder(f, transform=preprocess, loader=_safe_loader)
                          for f in folders])
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == 'cuda')

    moments = RunningMoments(spec.dim)
    print(f"Extracting {len(data)} images with {spec.name} on {device} ({'l2-normalized' if l2 else 'raw'} features)")
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            out = spec.forward(model, images.to(device))
            if l2:
                out = F.normalize(out, p=2, dim=1)
            moments.update(out.cpu().numpy())
            print(f"  {100 * (i + 1) / len(loader):5.1f}%", end='\r')
    return moments


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--backbone', required=True, choices=list(BACKBONES))
    source = p.add_mutually_exclusive_group()
    source.add_argument('--data-root', default=os.path.join(DATASETS_DIR, 'imagenet21k'),
                        help="ImageNet-21K-P folder (default: datasets/imagenet21k)")
    source.add_argument('--moments', help="refit from a moments file saved by a previous run")
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    add_l2_argument(p)
    args = p.parse_args()
    l2 = resolve_l2(args.backbone, args.l2)

    out = pca_path(args.backbone, l2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.moments:
        other = 'raw' if l2 else 'l2'
        if args.moments.endswith(f'_{other}.npz'):
            p.error(f"{args.moments} holds {other} features; pass {'--no-l2' if l2 else '--l2'} to refit it")
        moments = RunningMoments.load(args.moments)
        if moments.sum.shape[0] != BACKBONES[args.backbone].dim:
            p.error(f"{args.moments} has dimension {moments.sum.shape[0]}, "
                    f"but {args.backbone} features have dimension {BACKBONES[args.backbone].dim}")
    else:
        if not os.path.isdir(args.data_root):
            p.error(f"ImageNet-21K-P folder not found: {args.data_root} (see datasets/README.md or use --data-root)")
        moments = accumulate_moments(args.backbone, args.data_root, l2, args.batch_size, args.num_workers)
        moments_file = moments_path(args.backbone, l2)
        moments.save(moments_file)
        print(f"\nSaved {moments_file}")

    mean, components, eigenvalues = fit_pca(moments)
    np.savez(out, mean=mean.astype(np.float32), components=components.astype(np.float32),
             eigenvalues=eigenvalues.astype(np.float32))
    print(f"Saved {out} (fitted on {moments.n} images)")


if __name__ == '__main__':
    main()
