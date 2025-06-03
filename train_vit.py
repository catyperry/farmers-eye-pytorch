import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Train a Vision Transformer on crop images.")
parser.add_argument('--data_dir_train', type=str, required=True)
parser.add_argument('--data_dir_test', type=str, required=False, default=None, help="Path to the input test data directory. If not provided, testing will be skipped.")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--output_model_path', type=str, default='vit_model.pth')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT models usually expect 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ViT weights often use this normalization
])

# --- Load dataset ---
train_dataset = datasets.ImageFolder(args.data_dir_train, transform=transform)
num_classes = len(train_dataset.classes)
print(f"Found {len(train_dataset)} images, {num_classes} classes: {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

# --- Load pretrained ViT ---
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
model = model.to(device)

# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# --- Training loop ---
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    acc = correct / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")


# --- Test evaluation ---
if args.data_dir_test:
    def evaluate(loader, model, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total
    test_dataset = datasets.ImageFolder(args.data_dir_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Found {len(test_dataset)} test images, {len(test_dataset.classes)} classes: {test_dataset.classes}")
    test_acc = evaluate(test_loader, model, device)
    print(f"Test accuracy: {test_acc:.4f}")

# --- Save model ---
torch.save(model.state_dict(), args.output_model_path)
print(f"Model saved to {args.output_model_path}")