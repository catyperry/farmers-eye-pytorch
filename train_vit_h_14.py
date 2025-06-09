import gc
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

writer = SummaryWriter()

class PreprocessedTensorDataset(Dataset):
    def __init__(self, tensor_root: str, to_device: torch.device):
        self.tensor_paths: list[str] = []
        self.to_device: torch.device = to_device

        # Load class index
        self.class_to_idx: dict[str, int] = torch.load(os.path.join(tensor_root, 'class_to_idx.pt'))
        self.classes: list[str] = sorted(self.class_to_idx, key=self.class_to_idx.get)

        # Collect all .pt files
        for class_name in self.classes:
            class_dir = os.path.join(tensor_root, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".pt"):
                    self.tensor_paths.append(os.path.join(class_dir, file))

    def __getitem__(self, index):
        tensor, label = torch.load(self.tensor_paths[index])
        if self.to_device:
            tensor: torch.Tensor = tensor.to(self.to_device, non_blocking=True)
            label: torch.Tensor = torch.tensor(label, device=self.to_device)
        return tensor, label

    def __len__(self):
        return len(self.tensor_paths)


def evaluate(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating"):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module, device: torch.device, epoch: int, num_epochs: int, optimizer: optim.Optimizer, criterion: nn.Module, scaler: GradScaler) -> float:
    model.train()
    running_loss = 0.0
    total_len = len(loader.dataset) # type: ignore
    iter_len = total_len // (loader.batch_size or 1)
    for images, labels in tqdm(loader, total=iter_len, desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / total_len
    return avg_loss


def save(output_model_dir: str, epoch:int, model: nn.Module, optimizer: optim.SGD, scaler: GradScaler):
    output_model_path = os.path.join(output_model_dir, f"{epoch:06d}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, output_model_path)
    print(f"Model saved to {output_model_path}")

def load(output_model_dir: str, model: nn.Module, optimizer: optim.SGD, scaler: GradScaler) -> int:
    output_model_path = os.path.join(output_model_dir, sorted(os.listdir(output_model_dir))[-1])
    checkpoint = torch.load(output_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] - 1
    print(f"Model loaded from {output_model_path}")
    return start_epoch

def main(data_dir_train: str, data_dir_test: str, output_model_dir: str, batch_size: int, num_epochs:int, learning_rate: float, resume: bool, test: bool):
    # --- 1. Settings ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # --- 2. Data transforms ---
    # done now in preprocessing

    # --- 3. Load training dataset ---
    train_dataset = PreprocessedTensorDataset(data_dir_train, to_device=device)
    num_classes = len(train_dataset.classes)
    print(f"Train Dataset: Found {len(train_dataset)} images, {num_classes} classes: {train_dataset.classes}")

    if test:
        test_dataset = PreprocessedTensorDataset(data_dir_test, to_device=device)
        print(f"Test Dataset: Found {len(test_dataset)} images, { len(test_dataset.classes)} classes: {test_dataset.classes}")

    # --- 4. Split dataset ---
    train_loader = DataLoader[PreprocessedTensorDataset](train_dataset, batch_size=batch_size, shuffle=True)
    if test:
        test_loader = DataLoader[PreprocessedTensorDataset](test_dataset, batch_size=batch_size, shuffle=True)

    # --- 5. Load pre-trained MobileNetV2 ---
    # same weights as in the paper would be: IMAGENET1K_V1 instead of DEFAULT
    model_name = "vit_huge_patch14_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    # Freeze all layers and replace the head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(model.head.parameters(), lr=learning_rate, momentum=0.0)
    scaler = GradScaler()
    start_epoch = int(0)

    if resume:
        start_epoch = load(output_model_dir, model, optimizer, scaler)

    print(f"Model to {device}")
    model = model.to(device)

    # --- 6. Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    print(f"Criterion to {device}")
    criterion.to(device)


    # --- 7. Training loop ---
    test_every_x_epochs = 20
    epoch = int(0)
    for epoch in range(start_epoch+1, num_epochs+1):
        avg_loss = train(train_loader, model, device, epoch, num_epochs, optimizer, criterion, scaler)
        writer.add_scalar('avg_loss', avg_loss, epoch)

        if test and (epoch % test_every_x_epochs) == 0:
            train_acc = evaluate(test_loader, model)
            writer.add_scalar('train_acc', train_acc, epoch)
            save(output_model_dir, epoch, model, optimizer, scaler)
            print(f"Test Accuracy={train_acc:.4f}")
    
    # --- 8. Test accuracy ---
    # dont rerun if it just ran
    if test and not (epoch % test_every_x_epochs) == 0:
        train_acc = evaluate(test_loader, model)
        writer.add_scalar('train_acc', train_acc, epoch)
        print(f"Test Accuracy={train_acc:.4f}")

    del train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # --- 9. Save model ---
    save(output_model_dir, epoch, model, optimizer, scaler)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model with custom data.")
    parser.add_argument('--output_model_dir', type=str, required=True, help="Path to the directory for the models to save (and resume).")
    parser.add_argument('--data_dir_train', type=str, required=True, help="Path to the input training data directory.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for training (default: 1000).")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs (default: 10).")
    parser.add_argument('--resume', type=bool, default=False, help="Resume the training.")
    parser.add_argument('--test', type=bool, default=False, help="Activate accuracy on balanced test data.")
    parser.add_argument('--data_dir_test', type=str, required=False, help="Path to the input test_balanced data directory.")
    args = parser.parse_args()

    # Conditional requirement check
    if args.test and not args.data_dir_test:
        parser.error("--data_dir_test is required when --test is set to True.")

    learning_rate = 0.0035148759
    main(args.data_dir_train, args.data_dir_test, args.output_model_dir, args.batch_size, args.num_epochs, learning_rate, args.resume, args.test)