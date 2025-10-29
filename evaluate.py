import argparse
import torch
import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from src.evaluation.evaluator import evaluate_model
from src.data_pipeline.data_loader import get_dataloaders

from src.models.base.resnet import ResNet50FeatureExtractor

def evaluate(device="cpu"):
    _, _, test_loader,num_classes, class_names = get_dataloaders(args.data_dir)
    feature_extractor = ResNet50FeatureExtractor()

    # Build classifier
    images, _ = next(iter(test_loader))
    feature_extractor.eval()
    with torch.no_grad():
        sample_features = feature_extractor(images)
    input_dim = sample_features.size(1)
    classifier = torch.nn.Linear(input_dim, num_classes)

    # Load weights
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "saved_model.pth")
    classifier.load_state_dict(torch.load(checkpoint_path))
    classifier.eval()

    # Run the evaluation
    evaluate_model(
        classifier, test_loader, feature_extractor, class_names,
        device=device,
        save_dir=os.path.join(project_root, "reports")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Dog Breed Classifier")
    parser.add_argument("--model_path", type=str, default="checkpoints/saved_model.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default="data/split")
    args = parser.parse_args()
    evaluate(device=args.device)
