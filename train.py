import gc
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from torchvision.models import MobileNet_V2_Weights

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


# --- Passing an argument for data_dir ---
def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain a model with custom data.")
    parser.add_argument('--data_dir_train', type=str, required=True, help="Path to the input training data directory.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for training (default: 1000).")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs (default: 10).")
    parser.add_argument('--test', type=bool, default=False, help="Activate accuracy on balanced test data.")
    parser.add_argument('--data_dir_test', type=str, required=False, help="Path to the input test_balanced data directory.")
    parser.add_argument('--output_model_path', type=str, required=True, help="Path to the model for save (and resume).")
    parser.add_argument('--resume', type=bool, default=False, help="Resume the training.")
    args = parser.parse_args()

    # Conditional requirement check
    if args.test and not args.data_dir_test:
        parser.error("--data_dir_test is required when --test is set to True.")

    return args


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
    total_len = len(loader.dataset)
    iter_len = total_len // loader.batch_size 
    for images, labels in tqdm(loader, total=iter_len):
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


def save(output_model_path: str, epoch:int, model: models.MobileNetV2, optimizer: optim.SGD, scaler: GradScaler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, output_model_path)
    print(f"Model saved to {output_model_path}")

def load(output_model_path: str, model: models.MobileNetV2, optimizer: optim.SGD, scaler: GradScaler) -> int:
    checkpoint = torch.load(output_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Model loaded from {output_model_path}")
    return start_epoch

def main(data_dir_train: str, data_dir_test: str, output_model_path: str, batch_size: int, num_epochs:int, learning_rate: float, resume: bool, test: bool):
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
    model: nn.Module = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    optimizer = optim.SGD(model.classifier[1].parameters(), lr=learning_rate, momentum=0.0) # Used in paper: Gradient descent
    scaler = GradScaler()
    start_epoch = int(0)

    if resume:
        start_epoch = load(output_model_path, model, optimizer, scaler)

    model = model.to(device)
    model.compile()

    # --- 6. Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    criterion.compile()


    # --- 7. Training loop ---
    test_every_x_epochs = 50
    epoch = int(0)
    for epoch in range(start_epoch, num_epochs):
        avg_loss = train(train_loader, model, device, epoch, num_epochs, optimizer, criterion, scaler)
        writer.add_scalar('avg_loss', avg_loss, epoch)

        if test and (epoch % test_every_x_epochs) == (test_every_x_epochs-1):
            train_acc = evaluate(test_loader, model)
            # print(f"Test Accuracy={train_acc:.4f}")
            writer.add_scalar('train_acc', train_acc, epoch)
            save(output_model_path, epoch, model, optimizer, scaler)
    
    # --- 8. Test accuracy ---
    if test == True:
        train_acc = evaluate(test_loader, model)
        writer.add_scalar('train_acc', train_acc, epoch)
        print(f"Test Accuracy={train_acc:.4f}")

    del train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # --- 9. Save model ---
    save(output_model_path, epoch, model, optimizer, scaler)


if __name__ == "__main__":    
    args = arg_parse()
    output_model_path = './outputs/model.pth'
    learning_rate = 0.0035148759
    main(args.data_dir_train, args.data_dir_test, args.output_model_path, args.batch_size, args.num_epochs, learning_rate, args.resume, args.test)