import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset


# ---- 1. Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- 2. Load full dataset ----
dataset = datasets.ImageFolder(root="dogs_dataset", transform=transform)
total_size = len(dataset)

# ---- 2a. Split into Train / Validation / Test ----
torch.manual_seed(42)  # קבוע כדי שהחלוקה תהיה תמיד אותה
indices = torch.randperm(total_size)

test_size = int(0.15 * total_size)
valid_size = int(0.15 * total_size)
train_size = total_size - valid_size - test_size

train_indices = indices[:train_size]
valid_indices = indices[train_size:train_size+valid_size]
test_indices = indices[train_size+valid_size:]

train_dataset = Subset(dataset, train_indices)
valid_dataset = Subset(dataset, valid_indices)
test_dataset  = Subset(dataset, test_indices)

# ---- Optional: לקחת subset קטן ל-fast experiment ----
subset_small = 3000
if len(train_dataset) > subset_small:
    torch.manual_seed(42)  # כדי שהבחירה תמיד תחזור על עצמה
    small_indices = torch.randperm(len(train_dataset))[:subset_small]
    train_dataset = Subset(train_dataset, small_indices)


# ---- 3. Create DataLoaders ----
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train images:", len(train_dataset))
print("Valid images:", len(valid_dataset))
print("Test images:", len(test_dataset))
print("Classes (breeds):", dataset.classes[:5], "...")

# ---- 4. Load pretrained ResNet50 ----
resnet = models.resnet50(weights="IMAGENET1K_V1")
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# ---- 5. Function to extract feature vectors ----
def extract_features(loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

# ---- 6. Extract features ----
train_features, train_labels = extract_features(train_loader)
valid_features, valid_labels = extract_features(valid_loader)
test_features, test_labels   = extract_features(test_loader)

# ---- 7. Define classifier ----
num_classes = len(dataset.classes)
classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)

# ---- 8. Loss + optimizer ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
batch_size = 32
num_epochs = 5
num_batches = (train_features.size(0) + batch_size - 1) // batch_size

for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        features = train_features[start:end]
        labels = train_labels[start:end]

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ---- Validation ----
    classifier.eval()
    correct, total = 0, 0
    num_valid_batches = (valid_features.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_valid_batches):
            start = i * batch_size
            end = start + batch_size
            features = valid_features[start:end]
            labels = valid_labels[start:end]
            outputs = classifier(features)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {total_loss / num_batches:.4f} | Valid Acc: {acc:.4f}")


# ---- 11. Test (final evaluation) ----
#with torch.no_grad():
 #   test_outputs = classifier(test_features)
  #  test_probs = nn.Softmax(dim=1)(test_outputs)  # אחוזים לכל גזע
   # test_predicted = torch.argmax(test_probs, dim=1)
    #test_accuracy = (test_predicted == test_labels).float().mean()
#print("Test Accuracy:", test_accuracy.item())