# src/models/base/efficientnet.py
from torchvision import models
from torch import nn
from .base_model import BaseFeatureExtractor

class EfficientNetB0FeatureExtractor(BaseFeatureExtractor):
    def _build_model(self):
        efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        return nn.Sequential(*list(efficientnet.children())[:-1])
