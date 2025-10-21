# src/models/base/base_model.py
import torch
import torch.nn as nn


class BaseFeatureExtractor(nn.Module):
    """
    Base class for feature extractor models.
    All feature extractor models should inherit from this class
    and implement the `_build_model()` method.
    """
    def __init__(self):
        super().__init__()
        self.model = self._build_model()
        self.model.eval()  # we don't train the backbone

    def _build_model(self):
        """
        Should be implemented by child classes.
        Must return a nn.Module containing the feature extractor (without classifier head)
        """
        raise NotImplementedError

    def forward(self, x):
        features = self.model(x)
        return features.view(features.size(0), -1)

    def extract_features(self, loader):
        """
        Extract features from a DataLoader
        """
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                features = self.forward(images)
                all_features.append(features)
                all_labels.append(labels)
        return torch.cat(all_features), torch.cat(all_labels)
