import argparse
import os
import torch
from src.data_pipeline.data_loader import get_dataloaders
from src.models.base.resnet import ResNet50FeatureExtractor
from src.training.trainer import train_classifier


def main():
    # ---------------------------
    # Parse arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="Train Dog Breed Classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # ---------------------------
    # Set project paths
    # ---------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "saved_model.pth")
    SPLIT_DIR = "data/split"

    # ---------------------------
    # Load data
    # ---------------------------
    train_loader, valid_loader, test_loader,num_classes, class_names = get_dataloaders(SPLIT_DIR,args.batch_size)

    # ---------------------------
    # Feature extraction
    # ---------------------------
    feature_extractor = ResNet50FeatureExtractor()
    print("Extracting features for training/validation...")

    train_features, train_labels = feature_extractor.extract_features(train_loader)
    valid_features, valid_labels = feature_extractor.extract_features(valid_loader)

    # ---------------------------
    # Build classifier head
    # ---------------------------
    input_dim = train_features.size(1)
    classifier = torch.nn.Linear(input_dim, num_classes)

    # ---------------------------
    # Train classifier
    # ---------------------------
    train_classifier(
        classifier=classifier,
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=REPORTS_DIR,
        model_path=MODEL_PATH,
    )


if __name__ == "__main__":
    main()
