import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from PIL import ImageFile
from torchvision.models import MobileNet_V2_Weights


class PreprocessedTensorDataset(Dataset):
    def __init__(self, tensor_root: str, to_device: torch.device):
        self.tensor_paths = []
        self.to_device = to_device

        # Load class index
        self.class_to_idx = torch.load(os.path.join(tensor_root, 'class_to_idx.pt'))
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)

        # Collect all .pt files
        for class_name in self.classes:
            class_dir = os.path.join(tensor_root, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".pt"):
                    self.tensor_paths.append(os.path.join(class_dir, file))

    def __getitem__(self, index):
        tensor, label = torch.load(self.tensor_paths[index])
        if self.to_device:
            tensor = tensor.to(self.to_device, non_blocking=True)
            label = torch.tensor(label, device=self.to_device)
        return tensor, label

    def __len__(self):
        return len(self.tensor_paths)


# --- Passing an argument for data_dir ---
def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain a model with custom data.")
    parser.add_argument('--data_dir_train', type=str, required=True, help="Path to the input training data directory.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for training (default: 1000).")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs (default: 10).")
    parser.add_argument('--test', type=bool, default=False, help="Activate accuracy on balanced test data.")
    parser.add_argument('--data_dir_test', type=str, required=False, help="Path to the input test_balanced data directory.")

    args = parser.parse_args()

    # Conditional requirement check
    if args.test and not args.data_dir_test:
        parser.error("--data_dir_test is required when --test is set to True.")

    return args


def evaluate(loader: DataLoader, model: models.MobileNetV2):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluation"):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train(loader: DataLoader, model: models.MobileNetV2, device: torch.device, epoch: int, num_epochs: int, optimizer: optim.SGD, criterion: nn.CrossEntropyLoss, scaler: GradScaler):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

def test(data_dir_test: str, batch_size: int, model: models.MobileNetV2, device: torch.device):
    print("Evaluating on balanced test set...")

    # --- 8a) Load test dataset ---
    test_dataset = PreprocessedTensorDataset(data_dir_test, to_device=device)
    num_classes = len(test_dataset.classes)
    print(f"Found {len(test_dataset)} images, {num_classes} classes: {test_dataset.classes}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # --- 8b) Evaluate ---
    test_acc = evaluate(test_loader, model)
    print(f"Balaced test accuracy: {test_acc:.4f}")

def main(data_dir_train: str, output_model_path: str, batch_size: int, num_epochs:int, learning_rate: float):
    # --- 1. Settings ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Data transforms ---
    # done now in preprocessing

    # --- 3. Load training dataset ---
    train_dataset = PreprocessedTensorDataset(data_dir_train, to_device=device)
    num_classes = len(train_dataset.classes)
    print(f"Found {len(train_dataset)} images, {num_classes} classes: {train_dataset.classes}")

    # --- 4. Split dataset ---
    #n_total = len(full_dataset)
    #n_test = int(test_percent * n_total)
    #n_validation = int(validation_percent * n_total)
    #n_train = n_total - n_validation - n_test
    #train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [n_train, n_validation, n_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- 5. Load pre-trained MobileNetV2 ---
    # same weights as in the paper would be: IMAGENET1K_V1 instead of DEFAULT
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    model.compile()

    # --- 6. Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    criterion.compile()
    #optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.classifier[1].parameters(), lr=learning_rate, momentum=0.0) # Used in paper: Gradient descent
    scaler = GradScaler()

    # --- 7. Training loop ---
    for epoch in range(num_epochs):
        train(train_loader, model, device, epoch, num_epochs, optimizer, criterion, scaler)

    train_acc = evaluate(train_loader, model)
    print(f"Training Accuracy={train_acc:.4f}")

    # --- 8. Test accuracy ---
    if args.test == True:
        test(args.data_dir_test, batch_size, model, device)

    # --- 9. Save model ---
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":    
    args = arg_parse()
    data_dir_train = args.data_dir_train  # your data path
    output_model_path = './outputs/model.pth'
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = 0.0035148759
    main(data_dir_train, output_model_path, batch_size, num_epochs, learning_rate)