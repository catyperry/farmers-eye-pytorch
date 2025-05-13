import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- 1. Settings ---
data_dir = '/content/drive/MyDrive/farmers_eye/inputs/training'  # your data path
output_model_path = '/content/drive/MyDrive/farmers_eye/outputs/model.pth'
batch_size = 32
num_epochs = 10
learning_rate = 0.001
val_pct = 0.1
test_pct = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. Data transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# --- 3. Load dataset ---
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)
print(f"Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

# --- 4. Split dataset ---
n_total = len(full_dataset)
n_test = int(test_pct * n_total)
n_val = int(val_pct * n_total)
n_train = n_total - n_val - n_test
train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# --- 5. Load pre-trained MobileNetV2 ---
model = models.mobilenet_v2(pretrained=True)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# --- 6. Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

# --- 7. Training loop ---
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

# --- 8. Test accuracy ---
test_acc = evaluate(test_loader)
print(f"Test accuracy: {test_acc:.4f}")

# --- 9. Save model ---
torch.save(model.state_dict(), output_model_path)
print(f"Model saved to {output_model_path}")
