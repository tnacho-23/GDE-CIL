# datasets/

Raw image datasets (not tracked by git):

```
datasets/
  cifar-100-python/                  torchvision CIFAR-100 (meta, train, test)
  caltech256/256_ObjectCategories/   torchvision Caltech-256, one folder per category
  imagenet-r/                        ImageNet-R, one folder per WordNet ID
  imagenet21k/                       ImageNet-21K-P, only for fit_pca.py
    imagenet21k_train/<class>/*.JPEG
    imagenet21k_val/<class>/*.JPEG
```

`imagenet21k/` may contain any number of folders, each with one sub-folder of
images per class; all images are used and labels are ignored.

Set `GDECIL_DATASETS_DIR` to use another location for all datasets, or pass
`--data-root` to fit_pca.py for ImageNet-21K-P only.
