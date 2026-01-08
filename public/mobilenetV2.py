import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# === Seed cố định để tái hiện ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# === Cấu hình ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
batch_size = 32
num_classes = 6
epochs = 80
lr = 1e-4
patience = 20  
save_path = "mobilenetv2_best_yolov8_tuned_3.pth"

train_dir = "D:/KLTN/FINAL_SKINTONE/public/dataset_13/dataset_yolov8/train"
val_dir   = "D:/KLTN/FINAL_SKINTONE/public/dataset_13/dataset_yolov8/val"
test_dir  = "D:/KLTN/FINAL_SKINTONE/public/dataset_13/dataset_yolov8/test"

# === Albumentations Transform (giảm phá ảnh) ===
train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    # A.ColorJitter(p=0.2),
    A.Rotate(limit=5, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# === Custom Dataset ===
class AlbumentationsImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = np.array(Image.open(path).convert("RGB"))
        if self.albumentations_transform:
            image = self.albumentations_transform(image=image)["image"]
        return image, label

# === Load dataset ===
train_dataset = AlbumentationsImageFolder(train_dir, transform=train_transform)
val_dataset   = AlbumentationsImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Load base model & fine-tune sâu hơn ===
base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

for param in list(base_model.parameters())[:-10]:  # Unfreeze sâu hơn
    param.requires_grad = False

base_model.classifier = nn.Sequential(
    nn.Dropout(0.2),    
    nn.Linear(base_model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, num_classes)
)
model = base_model.to(device)

# === Loss, optimizer, scheduler ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

# === Train/Validation function ===
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), all_preds, all_labels

# === Training loop ===
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'lr': []}
best_acc = 0.0
early_stop_counter = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
    scheduler.step(val_loss)

    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"  Best model saved at epoch {epoch+1}")
    else:
        early_stop_counter += 1
        print(f" ⏳ No improvement. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

# === Load best model & evaluate ===
model.load_state_dict(torch.load(save_path))
model.eval()
val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
print(f"\n Best Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

# === Save history ===
with open("mobilenetv2_best_yolov8_tuned_5.json", "w") as f:
    json.dump(history, f)

# === Plot history ===
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.legend(); plt.title('LR Schedule')

plt.tight_layout(); plt.show()

# === Classification report (val) ===
print("\n Classification Report (Validation):")
print(classification_report(val_labels, val_preds, target_names=val_dataset.classes))

# === Đánh giá TEST set ===
if os.path.exists(test_dir):
    print("\n Đánh giá mô hình trên tập TEST...")
    test_dataset = AlbumentationsImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion)

    print(f"\n Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))

    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.show()

    pd.DataFrame({
        'True': [test_dataset.classes[i] for i in test_labels],
        'Pred': [test_dataset.classes[i] for i in test_preds]
    }).to_csv("test_predictions_albu.csv", index=False)
else:
    print("  Không tìm thấy thư mục test.")
