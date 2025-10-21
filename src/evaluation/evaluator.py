import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os
from src.visualization.plots import (
    plot_classification_report_heatmap, plot_confusion_matrix,
    plot_top_confusion_heatmap, plot_top_confusions_bar
)

def evaluate_model(classifier, test_loader, feature_extractor, classes, device="cpu", save_dir="reports"):
    figures_dir = os.path.join(save_dir, "figures")
    tables_dir = os.path.join(save_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = feature_extractor(images).view(images.size(0), -1)
            outputs = classifier(features)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.3f} ({correct}/{total})")
    with open(os.path.join(tables_dir, "test_accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.3f} ({correct}/{total})\n")

    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    metrics_only = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    sorted_metrics = metrics_only.sort_values(by="f1-score", ascending=False)
    final_report = pd.concat([sorted_metrics, report_df.loc[["macro avg", "weighted avg"]]], axis=0)
    print("\nClassification Report (sorted by F1):")
    print(final_report.round(3))
    final_report.to_csv(os.path.join(tables_dir, "classification_report.csv"))

    # Plots
    plot_classification_report_heatmap(final_report, classes, os.path.join(figures_dir, "classification_report.png"))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(figures_dir, "confusion_matrix.png"))
    plot_top_confusion_heatmap(cm, classes, top_n=10, save_path=os.path.join(figures_dir, "confusion_top10.png"))
    plot_top_confusions_bar(cm, classes, top_n=15, save_path=os.path.join(figures_dir, "top15_confusions.png"))

    print(f"📁 Plots saved in {figures_dir}")
