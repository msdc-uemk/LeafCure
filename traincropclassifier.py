import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import warnings

# -----------------------------
# Suppress warnings
# -----------------------------
warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================
data_dir = "dataset/cropclassification"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
batch_size = 16   # 🔥 reduced from 32 → helps with CUDA OOM
num_epochs = 20
num_classes = 3  # bellpepper, potato, tomato
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

# =============================
# DATA TRANSFORMS
# =============================
transform = {
    "train": transforms.Compose([
        transforms.Resize((300, 300)),  # EfficientNet-B3 prefers larger images
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(train_dir, transform=transform["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=transform["val"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train images: {len(train_dataset)}, Validation images: {len(val_dataset)}")
print(f"Classes found: {train_dataset.classes}")

# =============================
# MODEL
# =============================
model = models.efficientnet_b3(weights='IMAGENET1K_V1')  # pretrained weights
num_features = model.classifier[1].in_features  # EfficientNet classifier layer
model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)

# =============================
# LOSS & OPTIMIZER
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # smaller LR for fine-tuning

# =============================
# TRAINING LOOP WITH CHECKS
# =============================
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-"*30)

    # ---- TRAIN ----
    model.train()
    running_loss, running_corrects = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)

    # ---- VALIDATION ----
    model.eval()
    val_loss, val_corrects = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # =============================
    # CHECKS
    # =============================
    if train_acc < 0.40 and epoch > 5:
        print("⚠️ Warning: Model may not be learning properly.")
    if abs(train_acc - val_acc) > 0.30:
        print("⚠️ Warning: Possible overfitting.")
    if val_acc > 0.90:
        print("✅ Model performing well!")

    # ---- Save best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "static/model/crop_classifier.pth")
        print(f"Best model saved with val acc: {best_val_acc:.4f}")

print("\nTraining complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
