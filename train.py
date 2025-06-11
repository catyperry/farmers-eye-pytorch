import gc
import os
import argparse
import json
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from models import MODEL_REGISTRY, MODEL_NAME

class PreprocessedTensorDataset(Dataset):
    def __init__(self, tensor_root: str, to_device: torch.device | None):
        self.tensor_paths: list[str] = []
        self.to_device = to_device

        # Load class index
        self.class_to_idx: dict[str, int] = torch.load(os.path.join(tensor_root, 'class_to_idx.pt'))
        self.classes: list[str] = sorted(self.class_to_idx, key=lambda x: self.class_to_idx[x])

        # Collect all .pt files
        for class_name in self.classes:
            class_dir = os.path.join(tensor_root, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".pt"):
                    self.tensor_paths.append(os.path.join(class_dir, file))

    def __getitem__(self, index):
        tensor, label = torch.load(self.tensor_paths[index])
        if self.to_device is not None:
            tensor: torch.Tensor = tensor.to(self.to_device, non_blocking=True)
            label: torch.Tensor = torch.tensor(label, device=self.to_device)
        return tensor, label

    def __len__(self):
        return len(self.tensor_paths)
    

writer = SummaryWriter()

def load_hyperparams(model_name: str, config_file: str | None = None, **overrides) -> Dict[str, Any]:
    """Load hyperparameters with precedence: CLI args > config file > model defaults"""
    
    # Start with model defaults
    hyperparams = MODEL_REGISTRY[model_name].get_default_hyperparams()
    
    # Override with config file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            file_config = json.load(f)
        hyperparams.update(file_config)
    
    # Override with CLI arguments (only non-None values)
    hyperparams.update({k: v for k, v in overrides.items() if v is not None})
    
    return hyperparams

def evaluate(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_epoch(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module, device: torch.device, 
                epoch: int, num_epochs: int, optimizer: optim.Optimizer, criterion: nn.Module, 
                scaler: GradScaler) -> float:
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

def save_checkpoint(output_model_dir: str, epoch: int, model: nn.Module, 
                   optimizer: optim.Optimizer, scaler: GradScaler):
    output_model_path = os.path.join(output_model_dir, f"{epoch:06d}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, output_model_path)
    print(f"Model saved to {output_model_path}")

def load_checkpoint(output_model_dir: str, model: nn.Module, 
                   optimizer: optim.Optimizer, scaler: GradScaler) -> int:
    output_model_path = os.path.join(output_model_dir, sorted(os.listdir(output_model_dir))[-1])
    checkpoint = torch.load(output_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Model loaded from {output_model_path}")
    return start_epoch

def main(model_name: MODEL_NAME, data_dir_train: str, data_dir_test: str | None, 
         output_model_dir: str, resume: bool, test: bool, 
         config_file: str | None = None, **hyperparams):
    
    # Load and merge hyperparameters
    final_hyperparams = load_hyperparams(model_name, config_file, **hyperparams)
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Validate paths exist
    if not os.path.exists(data_dir_train):
        raise FileNotFoundError(f"Training data directory not found: {data_dir_train}")
    
    # Get model configuration
    model_config = MODEL_REGISTRY[model_name]
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {final_hyperparams}")
    torch.cuda.empty_cache()

    # Load datasets
    train_dataset = PreprocessedTensorDataset(data_dir_train, to_device=device)
    num_classes = len(train_dataset.classes)
    print(f"Train Dataset: Found {len(train_dataset)} images, {num_classes} classes: {train_dataset.classes}")

    test_dataset: PreprocessedTensorDataset | None = None
    if test and data_dir_test:
        test_dataset = PreprocessedTensorDataset(data_dir_test, to_device=device)
        print(f"Test Dataset: Found {len(test_dataset)} images, {len(test_dataset.classes)} classes: {test_dataset.classes}")

    # Create data loaders
    train_loader = DataLoader[PreprocessedTensorDataset](
        train_dataset, 
        batch_size=final_hyperparams['batch_size'], 
        shuffle=True
    )
    test_loader: DataLoader[PreprocessedTensorDataset] | None = None
    if test and test_dataset is not None:
        test_loader = DataLoader[PreprocessedTensorDataset](
            test_dataset, 
            batch_size=final_hyperparams['batch_size'], 
            shuffle=False
        )

    # Create model and optimizer
    model = model_config.create_model(num_classes)
    optimizer = model_config.create_optimizer(model, final_hyperparams['learning_rate'])
    scaler = GradScaler()
    start_epoch = 0

    # Load checkpoint if resuming
    if resume:
        start_epoch = load_checkpoint(output_model_dir, model, optimizer, scaler)

    # Move model to device
    print(f"Moving model to {device}")
    model = model.to(device)
    
    # Compile model if supported
    if hasattr(model, 'compile'):
        print("Compiling model")
        model.compile()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if hasattr(criterion, 'compile'):
        criterion.compile()

    # Training loop
    test_every_x_epochs = final_hyperparams['test_every_x_epochs']
    
    for epoch in range(start_epoch + 1, final_hyperparams['num_epochs'] + 1):
        avg_loss = train_epoch(train_loader, model, device, epoch, 
                              final_hyperparams['num_epochs'], optimizer, criterion, scaler)
        writer.add_scalar('avg_loss', avg_loss, epoch)

        # Test and save periodically
        if test and test_loader is not None and (epoch % test_every_x_epochs) == 0:
            test_acc = evaluate(test_loader, model)
            writer.add_scalar('test_acc', test_acc, epoch)
            save_checkpoint(output_model_dir, epoch, model, optimizer, scaler)
            print(f"Test Accuracy={test_acc:.4f}")
    
    # Final test if not just done
    final_epoch = final_hyperparams['num_epochs']
    if test and test_loader is not None and (final_epoch % test_every_x_epochs) != 0:
        test_acc = evaluate(test_loader, model)
        writer.add_scalar('test_acc', test_acc, final_epoch)
        print(f"Final Test Accuracy={test_acc:.4f}")

    # Cleanup
    del train_loader, train_dataset
    if test_dataset:
        del test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Save final model
    save_checkpoint(output_model_dir, final_epoch, model, optimizer, scaler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with custom data.")
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODEL_REGISTRY.keys()),
                       help="Model architecture to use")
    
    # Data paths
    parser.add_argument('--data_dir_train', type=str, required=True,
                       help="Path to the input training data directory.")
    parser.add_argument('--data_dir_test', type=str,
                       help="Path to the input test data directory.")
    parser.add_argument('--output_model_dir', type=str, required=True,
                       help="Path to the directory for the models to save (and resume).")
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       help="JSON config file with hyperparameters")
    
    # Hyperparameter overrides
    parser.add_argument('--learning_rate', type=float,
                       help="Learning rate override")
    parser.add_argument('--batch_size', type=int,
                       help="Batch size override")
    parser.add_argument('--num_epochs', type=int,
                       help="Number of epochs override")
    parser.add_argument('--test_every_x_epochs', type=int,
                       help="Test frequency override")
    
    # Training options
    parser.add_argument('--resume', action='store_true',
                       help="Resume training from checkpoint")
    parser.add_argument('--test', action='store_true',
                       help="Enable testing during training")
    
    args = parser.parse_args()
    
    # Validate test requirements
    if args.test and not args.data_dir_test:
        parser.error("--data_dir_test is required when --test is used")
    
    # Extract hyperparameter overrides
    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'test_every_x_epochs': args.test_every_x_epochs,
    }
    
    main(
        model_name=args.model,
        data_dir_train=args.data_dir_train,
        data_dir_test=args.data_dir_test,
        output_model_dir=args.output_model_dir,
        resume=args.resume,
        test=args.test,
        config_file=args.config,
        **hyperparams
    )