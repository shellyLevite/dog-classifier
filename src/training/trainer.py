import os
import torch
from src.visualization.plots import plot_metrics


def train_classifier(classifier, train_features, train_labels, valid_features, valid_labels,
                     num_epochs=5, batch_size=32, lr=0.001,
                     save_dir="reports/figures", model_path="checkpoints/saved_model.pth"):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    num_batches = (train_features.size(0) + batch_size - 1) // batch_size
    train_losses, valid_accuracies = [], []
    best_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            features, labels = train_features[start:end], train_labels[start:end]

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validation ---
        classifier.eval()
        correct, total = 0, 0
        num_valid_batches = (valid_features.size(0) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_valid_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                features, labels = valid_features[start:end], valid_labels[start:end]
                outputs = classifier(features)
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / num_batches
        valid_acc = correct / total

        train_losses.append(train_loss)
        valid_accuracies.append(valid_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        # --- Save best model ---
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(classifier.state_dict(), model_path)
            print(f" Saved new best model to {model_path} (val acc = {valid_acc:.4f})")

    # Save training curves
    plot_metrics(train_losses, valid_accuracies, save_path=os.path.join(save_dir, "training_metrics.png"))
