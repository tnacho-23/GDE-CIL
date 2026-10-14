"""Frozen backbones.

Each backbone defines how to load the model with its preprocessing transform and
how to turn a batch of images into a (B, dim) feature tensor. `l2_normalize` is
the default of the --l2 / --no-l2 flag of the scripts.

torch, timm and transformers are imported inside the functions, so the
classifiers can be used without them.
"""
from dataclasses import dataclass
from typing import Callable


def _imagenet_transform(resize, crop=None):
    import torchvision.transforms as T
    steps = [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC)]
    if crop is not None:
        steps.append(T.CenterCrop(crop))
    steps += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(steps)


def _load_dinov2(device):
    import torch
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
    return model, _imagenet_transform(224, 224)


def _load_vitb(device):
    import torchvision.transforms as T
    from transformers import ViTImageProcessor, ViTModel
    name = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(name)
    model = ViTModel.from_pretrained(name).to(device).eval()
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return model, preprocess


def _load_vitm(device):
    import timm
    model = timm.create_model('vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k',
                              pretrained=True, num_classes=0).to(device).eval()
    config = timm.data.resolve_model_data_config(model)
    return model, timm.data.create_transform(**config, is_training=False)


def _forward_pooled(model, images):
    """Models whose forward pass already returns the pooled embedding."""
    return model(images)


def _forward_cls_token(model, images):
    """Hugging Face ViT: [CLS] token of the last hidden state."""
    return model(images).last_hidden_state[:, 0, :]


@dataclass(frozen=True)
class Backbone:
    name: str
    dim: int
    l2_normalize: bool
    load: Callable      # device -> (model, preprocess)
    forward: Callable   # (model, images) -> (B, dim) tensor


BACKBONES = {
    'dinov2': Backbone('DINOv2 ViT-B/14', 768, False, _load_dinov2, _forward_pooled),
    'vitb': Backbone('ViT-B/16 (ImageNet-21k)', 768, False, _load_vitb, _forward_cls_token),
    'vitm': Backbone('ViT-M/16 (timm vit_mediumd)', 512, False, _load_vitm, _forward_pooled),
}
