import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Passing an argument for data_dir ---

parser = argparse.ArgumentParser(description="Retrain a model with custom data.")
parser.add_argument('--data_dir_train', type=str, required=True, help="Path to the input training data directory.")
parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for training (default: 1000).")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs (default: 10).")
parser.add_argument('--test85', type=bool, default=False, help="Activate accuracy on balanced (85) test data.")
parser.add_argument('--data_dir_test85', type=str, required=False, help="Path to the input test85 data directory.")


args = parser.parse_args()

# Conditional requirement check
if args.test85 and not args.data_dir_test85:
    parser.error("--data_dir_test85 is required when --test is set to True.")

# --- 1. Settings ---
data_dir_train = args.data_dir_train  # your data path
output_model_path = '/content/drive/MyDrive/farmers_eye/outputs/model.pth'
batch_size = args.batch_size # Previously was 32
num_epochs = args.num_epochs
learning_rate = 0.0035148759
validation_percent = 0.1
test_percent = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. Data transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# --- 3. Load training dataset ---
train_dataset = datasets.ImageFolder(data_dir_train, transform=transform)
num_classes = len(train_dataset.classes)
print(f"Found {len(train_dataset)} images, {num_classes} classes: {train_dataset.classes}")




# --- 4. Split dataset ---
#n_total = len(full_dataset)
#n_test = int(test_percent * n_total)
#n_validation = int(validation_percent * n_total)
#n_train = n_total - n_validation - n_test
#train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [n_train, n_validation, n_test])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
#test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
#optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)
optimizer = optim.SGD(model.classifier[1].parameters(), lr=learning_rate, momentum=0.0) # Used in paper: Gradient descent


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
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

train_acc = evaluate(train_loader)
print(f"Training Accuracy={train_acc:.4f}")

# --- 8. Test accuracy ---
if args.test85 == True:
    print("Evaluating on balanced test set...")
    data_dir_test85 = args.data_dir_test85

    # --- 8a) Load test85 dataset ---
    test85_dataset = datasets.ImageFolder(data_dir_test85, transform=transform)
    num_classes = len(test85_dataset.classes)
    print(f"Found {len(test85_dataset)} images, {num_classes} classes: {test85_dataset.classes}")

    test85_loader = DataLoader(test85_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- 8b) Evaluate ---
    test85_acc = evaluate(test85_loader)
    print(f"Balaced test accuracy: {test85_acc:.4f}")

# --- 9. Save model ---
torch.save(model.state_dict(), output_model_path)
print(f"Model saved to {output_model_path}")
