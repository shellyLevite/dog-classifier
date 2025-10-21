import torch
BATCH_SIZE = 32
NUM_EPOCHS = 5
SUBSET_SMALL = 3000
SEED = 42
DATA_DIR = r"C:\Users\levit\OneDrive\שולחן העבודה\PROJECTX\dog-classifier\data\raw\dogs_dataset"
LR = 0.001
CHECKPOINT_PATH = "checkpoints/saved_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
