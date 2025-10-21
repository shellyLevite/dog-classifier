import os
import shutil
import random
import hashlib
import argparse
from tqdm import tqdm


def normalize_class_name(name: str):
    return name.lower().replace(" ", "_")

def file_hash(path: str):
    """Return a hash of the file content for duplicate detection."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def collect_images(raw_dir):
    """Collect all images from raw_dir/* and remove duplicates."""
    seen_hashes = set()
    class_images = {}

    for folder in os.listdir(raw_dir):
        images_folder = os.path.join(raw_dir, folder)
        if not os.path.isdir(images_folder):
            continue

        for cls in os.listdir(images_folder):
            cls_path = os.path.join(images_folder, cls)
            if not os.path.isdir(cls_path):
                continue
            normalized_cls = normalize_class_name(cls)
            class_images.setdefault(normalized_cls, [])

            for img_file in os.listdir(cls_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(cls_path, img_file)
                h = file_hash(img_path)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                class_images[normalized_cls].append(img_path)

    return class_images

def split_class_images(images, train_ratio, val_ratio, test_ratio):
    """Return dictionary of split images per split type."""
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

def copy_images_to_splits(class_images, split_dir, train_ratio, val_ratio, test_ratio):
    """Copy images to train/val/test directories."""
    for cls, images in tqdm(class_images.items(), desc="Splitting classes"):
        splits = split_class_images(images, train_ratio, val_ratio, test_ratio)
        for split, split_images in splits.items():
            dest_dir = os.path.join(split_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, os.path.join(dest_dir, os.path.basename(img)))


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Root directory with raw dataset")
    parser.add_argument("--split-dir", type=str, default="data/split", help="Destination directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    class_images = collect_images(args.raw_dir)
    copy_images_to_splits(class_images, args.split_dir, args.train_ratio, args.val_ratio, args.test_ratio)

    print("Dataset split completed!")

if __name__ == "__main__":
    main()