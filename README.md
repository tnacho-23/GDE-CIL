# GDE-CIL: Gaussian Density Estimation for Class-Incremental Learning

Code for *Revisiting Class-Incremental Learning in the Era of Foundation Models: A Gaussian Density Estimation Perspective* (Romero, Loncomilla and Ruiz del Solar, TMLR 2026, [OpenReview](https://openreview.net/forum?id=k3uhZV6lfU)).

Images are encoded by a frozen backbone and each class is modeled as a Gaussian, or a mixture of Gaussians, in that feature space. Learning a new class only means estimating its statistics: there is no backpropagation, replay buffer or exemplar storage. Prediction is MAP inference over the class densities with a uniform prior.

## Layout

```
extract_features.py   extract backbone features for a dataset
fit_pca.py            fit a PCA projection on ImageNet-21K-P features (optional)
train.py              train and evaluate a class-incremental classifier
gdecil/
  backbones.py        DINOv2-B/14, ViT-B/16, ViT-M/16
  datasets.py         task protocols and image sources
  features.py         feature files, train/test split, PCA reduction, task orders
  pca.py              running moments and PCA
  protocol.py         class-incremental evaluation and metrics
  results.py          accuracy matrices to .xlsx
  classifiers/        Gaussian classifiers (see below)
datasets/  features/  results/    data, extracted features and outputs (not tracked)
```

## Installation

```bash
pip install -r requirements.txt
```

A GPU is only useful for feature extraction; the classifiers run on CPU.

## Data

Put the raw datasets in `datasets/` (see `datasets/README.md`); ImageNet-21K-P is only needed for the PCA. The locations of datasets, features and results can be changed with the `GDECIL_DATASETS_DIR`, `GDECIL_FEATURES_DIR` and `GDECIL_RESULTS_DIR` environment variables.

| `--dataset` | Classes | Tasks | Images used |
|---|---|---|---|
| `cifar100` | 100 | 10 x 10 | CIFAR-100 training set |
| `imagenetr` | 200 | 10 x 20 | ImageNet-R, first 200 classes |
| `caltech256` | 250 | 10 x 25 | Caltech-256, first 250 classes |
| `joint` | 550 | 30 | the three above, tasks in random order |

Each class is split 80/20 into train and test (seed 42). The training part of every class is truncated to the size of the smallest class of its dataset, so all classes have the same number of training samples. Files of each class are read in sorted order.

## Backbones

| `--backbone` | Model | Dim | l2-normalized by default |
|---|---|---|---|
| `dinov2` | DINOv2 ViT-B/14 (`torch.hub facebookresearch/dinov2`) | 768 | no |
| `vitb` | ViT-B/16 ImageNet-21k (`google/vit-base-patch16-224-in21k`) | 768 | no |
| `vitm` | ViT-M/16 (`timm vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k`) | 512 | no |

## Feature extraction

```bash
python extract_features.py --backbone dinov2 --dataset all
python extract_features.py --backbone vitb --dataset cifar100
```

`--l2` l2-normalizes the features and `--no-l2` keeps the raw model output. Without either flag the backbone default in the table above is used. Features are written to `features/{backbone}/{dataset}_l2.npz` or `features/{backbone}/{dataset}_raw.npz`, so both versions can coexist:

```bash
python extract_features.py --backbone dinov2 --dataset all --l2      # normalized DINOv2 features
python extract_features.py --backbone vitb --dataset all --l2        # normalized ViT-B features
```

`train.py` and `fit_pca.py` take the same `--l2` / `--no-l2` flag (with the same default) to choose which features to use.

## Classifiers

```bash
python train.py --method <method> --dataset <dataset> [--backbone dinov2] [--runs 1] [--l2 | --no-l2]
```

| `--method` | Classifier | Options |
|---|---|---|
| `diag` | Per-class diagonal covariance (Diag-Cov) | `--eps`, `--no-norm` |
| `full` | Per-class full covariance (Full-Cov) | `--eps`, `--no-norm` |
| `shared_diag` | Diagonal covariance shared by all classes (Shared-Diag-Cov) | |
| `shared_full` | Full covariance shared by all classes, OAS (Shared-Full-Cov) | |
| `mog_full` | Per-class mixture of full-covariance Gaussians, BIC-selected (MOG4-F) | `--max-components`, `--no-norm` |
| `mog_shared` | Per-class mixture of Gaussians with one shared covariance (MOG4-Shared) | `--max-components` |
| `oas` | Per-class OAS shrinkage covariance | `--no-norm` |
| `lw_identity` | Per-class Ledoit-Wolf shrinkage towards a scaled identity | `--no-norm` |
| `lw_diag` | Per-class Ledoit-Wolf shrinkage towards the diagonal | `--no-norm` |

`--eps` is the covariance regularizer (default 1e-6), `--max-components` the largest number of mixture components tried by BIC (default 4) and `--no-norm` drops the Gaussian normalization term so classes are ranked by Mahalanobis distance only.

The classifier is fitted once and evaluated on `--runs` random task orders. For each run, the accuracy matrix R (R[k, j] = accuracy on task j after learning tasks 0..k) is saved to `results/{backbone}/{method}_{dataset}.xlsx` (non-default settings are appended to the name, e.g. `full_imagenetr_pca130.xlsx` or `diag_cifar100_l2.xlsx`), and the final average accuracy A_T and forgetting F_T are printed.

## PCA

Features can be reduced with a PCA fitted on ImageNet-21K-P (the processed ImageNet-21K release by Ridnik et al., 2021). Only the images are used, not the labels.

**1. Put ImageNet-21K-P in `datasets/imagenet21k/`.** It must contain one or more folders with one sub-folder of images per class, which is the layout of the official release:

```
datasets/imagenet21k/
  imagenet21k_train/n00005787/*.JPEG
  imagenet21k_val/n00005787/*.JPEG
  ...
```

If it is stored elsewhere, pass `--data-root /path/to/imagenet21k` in the next step.

**2. Fit the PCA** for a backbone (this runs the backbone over every image, so a GPU is recommended):

```bash
python fit_pca.py --backbone dinov2
```

The features are normalized with the same `--l2` / `--no-l2` rule as in `extract_features.py` (raw by default). The PCA must be fitted with the same normalization as the features it will reduce; fit it twice to have both. This writes, with suffix `_l2` or `_raw`:

| File | Content |
|---|---|
| `features/dinov2/pca_imagenet21k_raw.npz` | `mean`, `components` (principal axes as columns, by decreasing variance), `eigenvalues` |
| `features/dinov2/imagenet21k_moments_raw.npz` | number of images, sum and sum of outer products of the features |

To refit the PCA without extracting again: `python fit_pca.py --backbone dinov2 --moments features/dinov2/imagenet21k_moments_raw.npz`.

**3. Train on reduced features** with `--pca-dim D`, which keeps the first `D` principal components (1 to the backbone dimension, 768 for DINOv2) of the PCA with the same normalization as the features. It works with every method and dataset:

```bash
python train.py --method diag --dataset cifar100 --pca-dim 130
python train.py --method full --dataset imagenetr --pca-dim 64 --runs 3
```

Results are saved with a `_pca{D}` suffix, e.g. `results/dinov2/diag_cifar100_pca130.xlsx`.

## Citation

```bibtex
@article{2026revisitingcil,
  title   = {Revisiting Class-Incremental Learning in the Era of Foundation Models: A Gaussian Density Estimation Perspective},
  author  = {Romero, Ignacio and Loncomilla, Patricio and Ruiz del Solar, Javier},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=k3uhZV6lfU}
}
```
