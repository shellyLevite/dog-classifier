# src/models/base/resnet.py
import torch
from torchvision import models
from torch import nn
from .base_model import BaseFeatureExtractor

class ResNet50FeatureExtractor(BaseFeatureExtractor):
    def _build_model(self):
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        return nn.Sequential(*list(resnet.children())[:-1])
