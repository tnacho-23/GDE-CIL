# features/

Extracted features (not tracked by git). The suffix says whether the features are
l2-normalized (`_l2`) or the raw model output (`_raw`):

```
features/{backbone}/{dataset}_{l2|raw}.npz             features (N, dim) float32, labels (N,) int64
features/{backbone}/pca_imagenet21k_{l2|raw}.npz       mean, components, eigenvalues (fit_pca.py)
features/{backbone}/imagenet21k_moments_{l2|raw}.npz   moments used to fit the PCA
```

Generate them with `extract_features.py` and `fit_pca.py`. Set `GDECIL_FEATURES_DIR` to use another location.
