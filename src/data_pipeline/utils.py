import os
import json
import torch

def save_split_indices(train_indices, valid_indices, test_indices, path="data/processed/split_indices.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "train": train_indices,
            "valid": valid_indices,
            "test": test_indices
        }, f)

def load_split_indices(path="data/processed/split_indices.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            split_indices = json.load(f)
        return split_indices["train"], split_indices["valid"], split_indices["test"]
    else:
        return None, None, None

def create_split_indices(total_size, seed=42, test_ratio=0.15, valid_ratio=0.15):
    torch.manual_seed(seed)
    indices = torch.randperm(total_size).tolist()
    test_size = int(test_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    train_size = total_size - valid_size - test_size

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices  = indices[train_size + valid_size:]
    return train_indices, valid_indices, test_indices
