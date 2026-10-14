"""Train and evaluate a class-incremental Gaussian classifier on frozen features.

Usage:
    python train.py --method diag --dataset cifar100
    python train.py --method mog_shared --dataset joint --backbone vitm --runs 3
    python train.py --method full --dataset imagenetr --pca-dim 130

The classifier is fitted once on the training split and evaluated on --runs
random task orders. For each run, R[k, j] (accuracy on task j after learning
tasks 0..k) is written to results/{backbone}/{method}_{dataset}[_options].xlsx,
and the final average accuracy A_T and forgetting F_T are printed.

--l2 / --no-l2 selects l2-normalized or raw features (default: raw);
they must have been extracted with the same flag.

--pca-dim D projects the features onto the first D principal axes of the PCA
fitted by fit_pca.py with the same normalization.
"""
import argparse

import numpy as np
import pandas as pd

from gdecil.backbones import BACKBONES
from gdecil.classifiers import METHODS, build, options_of
from gdecil.datasets import DATASETS
from gdecil.features import add_l2_argument, load_pca, load_split, project, resolve_l2
from gdecil.protocol import average_accuracy, forgetting, run_protocol
from gdecil.results import results_path, save_results


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog='methods:\n' + '\n'.join(f'  {k:12s} {v[2]}' for k, v in METHODS.items()))
    p.add_argument('--method', required=True, choices=list(METHODS))
    p.add_argument('--dataset', required=True, choices=list(DATASETS))
    p.add_argument('--backbone', default='dinov2', choices=list(BACKBONES))
    p.add_argument('--runs', type=int, default=1, help="number of random task orders (default: 1)")
    p.add_argument('--pca-dim', type=int, help="reduce the features to this many PCA dimensions")
    p.add_argument('--eps', type=float, help="covariance regularizer of diag and full (default: 1e-6)")
    p.add_argument('--max-components', type=int, help="maximum mixture components for mog_* (default: 4)")
    p.add_argument('--no-norm', dest='use_norm_term', action='store_false', default=None,
                   help="drop the Gaussian normalization term (rank classes by Mahalanobis distance)")
    add_l2_argument(p)
    args = p.parse_args()
    l2 = resolve_l2(args.backbone, args.l2)
    if args.runs < 1:
        p.error("--runs must be at least 1")

    options = {'eps': args.eps, 'max_components': args.max_components, 'use_norm_term': args.use_norm_term}
    flags = {'eps': '--eps', 'max_components': '--max-components', 'use_norm_term': '--no-norm'}
    for name, value in options.items():
        if value is not None and name not in options_of(args.method):
            p.error(f"{flags[name]} does not apply to --method {args.method}")
    classifier = build(args.method, **options)

    try:
        X_train, y_train, X_test, y_test = load_split(args.backbone, args.dataset, l2)
        if args.pca_dim is not None:
            mean, components = load_pca(args.backbone, l2)
            if not 1 <= args.pca_dim <= components.shape[1]:
                p.error(f"--pca-dim must be between 1 and {components.shape[1]}")
            X_train, X_test = (project(X, mean, components, args.pca_dim) for X in (X_train, X_test))
    except FileNotFoundError as e:
        p.exit(1, f"error: {e}\n")
    print(f"{args.method} | {args.backbone} ({'l2' if l2 else 'raw'}) | {args.dataset} | "
          f"train {X_train.shape} | test {X_test.shape}")

    matrices = run_protocol(classifier, (X_train, y_train, X_test, y_test), args.dataset, args.runs)

    extra = {}
    if hasattr(classifier, 'n_components'):
        extra['Components'] = pd.DataFrame({'Class': np.arange(len(classifier.n_components)),
                                            'N_Components': classifier.n_components})
    tags = [] if l2 == resolve_l2(args.backbone) else ['l2' if l2 else 'raw']
    tags += [f'pca{args.pca_dim}'] if args.pca_dim is not None else []
    tags += [f'eps{args.eps:g}'] * (args.eps is not None) + [f'k{args.max_components}'] * (args.max_components is not None)
    tags += ['nonorm'] * (args.use_norm_term is False)
    path = results_path(args.backbone, args.method, args.dataset, tags)
    save_results(path, matrices, extra)

    acc = np.array([average_accuracy(R) for R in matrices]) * 100
    fgt = np.array([forgetting(R) for R in matrices]) * 100
    print(f"\nA_T = {acc.mean():.2f} +- {acc.std():.2f} %   F_T = {fgt.mean():.2f} +- {fgt.std():.2f} %")
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
