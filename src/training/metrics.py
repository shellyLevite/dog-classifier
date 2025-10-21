import torch

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).sum().item() / len(labels)
