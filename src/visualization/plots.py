import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metrics(train_losses, valid_accuracies, save_path="training_metrics.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # גרף ראשון - Train Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, color='tab:blue', marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # גרף שני - Validation Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:green')
    ax2.plot(epochs, valid_accuracies, color='tab:green', marker='x', label='Valid Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # כותרת לפני tight_layout עם padding כדי לא להיחתך
    plt.title("Training Metrics", pad=20)

    # סידור פריסה ושמירה
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_classification_report_heatmap(report_df, classes, save_path="classification_report.png"):
    plt.figure(figsize=(10, len(classes) * 0.4 + 2))
    sns.heatmap(report_df.iloc[:-3, :-1], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    plt.title("Precision / Recall / F1 per Class")
    plt.ylabel("Class")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    plt.figure(figsize=(max(12, len(classes) * 0.5), max(10, len(classes) * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top_confusion_heatmap(cm, classes, top_n=10, save_path="confusion_top10.png"):
    error_counts = cm.sum(axis=1) - np.diag(cm)
    top_idx = np.argsort(error_counts)[-top_n:]
    top_classes = [classes[i] for i in top_idx]
    cm_top = cm[np.ix_(top_idx, top_idx)]
    plt.figure(figsize=(max(8, top_n * 0.8), max(6, top_n * 0.6)))
    sns.heatmap(cm_top, annot=True, fmt="d", xticklabels=top_classes, yticklabels=top_classes, cmap="Reds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Top {top_n} Confused Classes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top_confusions_bar(cm, classes, top_n=15, save_path="top15_confusions.png"):
    errors = [(classes[i], classes[j], cm[i, j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    df_top = pd.DataFrame(errors, columns=["True", "Predicted", "Count"]).sort_values("Count", ascending=False).head(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Count", y=df_top.apply(lambda x: f"{x['True']} → {x['Predicted']}", axis=1), data=df_top, palette="viridis")
    plt.xlabel("Number of Mistakes")
    plt.ylabel("True → Predicted")
    plt.title(f"Top {top_n} Confusions Globally")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
