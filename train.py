import gc
import os
import argparse
import json
from typing import Dict, Any
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import csv
import pandas as pd

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

# --- Path Constants for Colab ---
# Source paths on Google Drive
COLAB_DRIVE_DATA_TRAIN_SRC = '/content/drive/MyDrive/farmers_eye/inputs/training'
COLAB_DRIVE_DATA_TEST_SRC = '/content/drive/MyDrive/farmers_eye/inputs/test_balanced'
# Destination paths on the fast local Colab runtime
COLAB_LOCAL_DATA_TRAIN_DEST = '/content/data_train_local'
COLAB_LOCAL_DATA_TEST_DEST = '/content/data_test_local'
# Persistent output directories on Google Drive
COLAB_DRIVE_MODELS_DIR = '/content/drive/MyDrive/farmers_eye/outputs'
COLAB_DRIVE_RUNS_DIR = '/content/drive/MyDrive/farmers_eye/runs'

def is_colab():
    """
    Check if running in Google Colab.
    This is more robust than checking sys.modules, as it works in shell subprocesses.
    """
    return 'COLAB_GPU' in os.environ

def setup_colab():
    """Verifies the Colab environment and checks for GPU (assumes Drive is pre-mounted)."""
    if is_colab():
        print("🔧 Verifying Google Colab environment...")
        if not os.path.exists('/content/drive/MyDrive'):
            print("⚠️ Google Drive is not mounted. Please mount it in a separate cell before running the script.")
        else:
            print("✅ Google Drive appears to be mounted.")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            print(f"🚀 GPU available: {gpu_name}")
        else:
            print("⚠️ No GPU available - training will be slow!")
        return True
    return False

def copy_data_to_local(source_dir: str, destination_dir: str):
    """Copy data from a source to a local destination if needed."""
    if not os.path.exists(source_dir):
        print(f"⚠️ Source directory not found, skipping copy: {source_dir}")
        return

    if os.path.exists(destination_dir):
        print(f"📁 Using existing local copy: {destination_dir}")
        return

    print(f"📋 Copying data to local storage for faster access...")
    print(f"   From: {source_dir}")
    print(f"   To:   {destination_dir}")
    try:
        import shutil
        import time
        start_time = time.time()
        shutil.copytree(source_dir, destination_dir)
        copy_time = time.time() - start_time
        print(f"✅ Data copied in {copy_time:.1f} seconds.")
    except Exception as e:
        print(f"⚠️ Failed to copy data locally: {e}")
        # If copy fails, we'll get a FileNotFoundError later, which is appropriate.

def get_default_paths():
    """Get default paths based on environment."""
    if is_colab():
        return {
            'data_dir_train': COLAB_LOCAL_DATA_TRAIN_DEST,
            'data_dir_test': COLAB_LOCAL_DATA_TEST_DEST,
            'output_model_dir': COLAB_DRIVE_MODELS_DIR,
            'runs_dir': COLAB_DRIVE_RUNS_DIR
        }
    else:
        return {
            'data_dir_train': './inputs/data_train_local',
            'data_dir_test': './inputs/data_test_local',
            'output_model_dir': './outputs',
            'runs_dir': './runs'
        }

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

def evaluate(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)
    
    accuracy = correct / total
    average_loss = running_loss / total
    return accuracy, average_loss

def train_epoch(loader: DataLoader[PreprocessedTensorDataset], model: nn.Module, device: torch.device, 
                epoch: int, num_epochs: int, optimizer: optim.Optimizer, criterion: nn.Module, 
                scaler: GradScaler) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
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
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / total_len
    accuracy = correct / total
    return avg_loss, accuracy

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
         config_file: str | None = None, runs_dir: str = './runs',
         use_local_copy: bool = True, metric: bool = False, optimizer_type: str = "adam",
        momentum: float = 0.0, **hyperparams):
    
    # Setup Colab if needed
    setup_colab()
    if is_colab() and use_local_copy:
        print("🔄 Optimizing data access for Colab...")
        copy_data_to_local(COLAB_DRIVE_DATA_TRAIN_SRC, data_dir_train)
        if test and data_dir_test:
            copy_data_to_local(COLAB_DRIVE_DATA_TEST_SRC, data_dir_test)
    
    # Load and merge hyperparameters
    final_hyperparams = load_hyperparams(model_name, config_file, **hyperparams)
    
    # Create output directories
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    csv_dir = os.path.join(output_model_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
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
    # Enhanced TensorBoard setup with unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(runs_dir, f'{model_name}_{timestamp}')
    writer = SummaryWriter(log_dir)
    writer.add_text('Hyperparameters', str(final_hyperparams))

    # Logging in .csv 
    metrics_log = []

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
    optimizer = model_config.create_optimizer(model, optimizer_type=optimizer_type, lr=final_hyperparams['learning_rate'], momentum=momentum)

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
    best_test_acc, train_loss, train_acc = 0.0, 0.0, 0.0

    print(f"\nStarting training from epoch {start_epoch + 1} to {final_hyperparams['num_epochs']}")
    print(f"Logging to TensorBoard directory: {log_dir}")
    
    if is_colab():
        print("📊 To view TensorBoard in Colab, run in a new cell:")
        print(f"%load_ext tensorboard")
        print(f"%tensorboard --logdir {runs_dir}")
    else:
        print("Run 'tensorboard --logdir=runs' to view training progress")
    print()

    # Test at Epoch 0 if testing is enabled
    if test and test_loader is not None and start_epoch == 0:
        test_acc_0, test_loss_0 = evaluate(test_loader, model, criterion, device)
        writer.add_scalar('Loss/Test', test_loss_0, 0)
        writer.add_scalar('Accuracy/Test', test_acc_0, 0)
        best_test_acc = test_acc_0
        print(f"Epoch 0 - Test Loss: {test_loss_0:.4f}, Test Acc: {test_acc_0:.4f}")
    

    for epoch in range(start_epoch + 1, final_hyperparams['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(train_loader, model, device, epoch, 
                              final_hyperparams['num_epochs'], optimizer, criterion, scaler)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch}/{final_hyperparams['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Test and save periodically
        if test and test_loader is not None and (epoch % test_every_x_epochs) == 0:
            test_acc, test_loss = evaluate(test_loader, model, criterion, device)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            
            # Track best accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                writer.add_scalar('Best_Test_Accuracy', best_test_acc, epoch)
            
            #save_checkpoint(output_model_dir, epoch, model, optimizer, scaler)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} (Best: {best_test_acc:.4f})")
        
        metrics_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss if test and (epoch % test_every_x_epochs == 0) else None,
            'test_acc': test_acc if test and (epoch % test_every_x_epochs == 0) else None,
            'best_test_acc': best_test_acc
        })  
    # Final test if not just done
    final_epoch = final_hyperparams['num_epochs']
    if test and test_loader is not None and (final_epoch % test_every_x_epochs) != 0:
        test_acc, test_loss = evaluate(test_loader, model, criterion, device)
        writer.add_scalar('Loss/Test', test_loss, final_epoch)
        writer.add_scalar('Accuracy/Test', test_acc, final_epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            writer.add_scalar('Best_Test_Accuracy', best_test_acc, final_epoch)
        print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} (Best: {best_test_acc:.4f})")

        metrics_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss if test and (epoch % test_every_x_epochs == 0) else None,
            'test_acc': test_acc if test and (epoch % test_every_x_epochs == 0) else None,
            'best_test_acc': best_test_acc
        }) 

    writer.add_hparams(
        hparam_dict={
            'model': model_name,
            'learning_rate': final_hyperparams['learning_rate'],
            'batch_size': final_hyperparams['batch_size'],
            'num_epochs': final_hyperparams['num_epochs'],
        },
        metric_dict={
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'best_test_acc': best_test_acc if test else 0.0,
        }
    )
    

    # Cleanup
    writer.close()

    # Save to .csv file
    if metric:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(csv_dir, f'metrics_{timestamp}_{model_name}.csv')
        # Convert metrics log to DataFrame
        metrics_df = pd.DataFrame(metrics_log)

        # Write hyperparams at the top manually
        with open(csv_file, 'w') as f:
            f.write(f'Metrics for {model_name} at {timestamp}\n')
            f.write(f'Optimizer, {optimizer_type}\n')
            if optimizer_type == "sgd":
                f.write(f'Momentum, {momentum}\n')
            f.write('Hyperparameters\n')
            for key, value in final_hyperparams.items():
                f.write(f'{key},{value}\n')
            f.write('\n')  # Blank line before metrics

        # Append metrics dataframe to the CSV
        metrics_df.to_csv(csv_file, mode='a', index=False)

    del train_loader, train_dataset
    if test_dataset:
        del test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Save final model
    save_checkpoint(output_model_dir, final_epoch, model, optimizer, scaler)

    print(f"\nTraining completed!")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    if is_colab():
        print("📊 View TensorBoard in Colab with:")
        print(f"%load_ext tensorboard")
        print(f"%tensorboard --logdir {runs_dir}")
    else:
        print(f"View with: tensorboard --logdir={runs_dir}")

if __name__ == "__main__":
    # Get default paths based on environment
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(description="Train models with custom data.")
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODEL_REGISTRY.keys()),
                       help="Model architecture to use")
    
    # Optimizer selection
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer to use (default: adam).")
    
    # Choose momentum, only for SGD
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Momentum for SGD optimizer (ignored if using Adam).")
    
    # Data paths with environment-aware defaults
    parser.add_argument('--data_dir_train', type=str, 
                       default=defaults['data_dir_train'],
                       help="Path to the input training data directory.")
    parser.add_argument('--data_dir_test', type=str,
                       default=defaults['data_dir_test'],
                       help="Path to the input test data directory.")
    parser.add_argument('--output_model_dir', type=str, 
                       default=defaults['output_model_dir'],
                       help="Path to the directory for the models to save (and resume).")
    parser.add_argument('--runs_dir', type=str,
                       default=defaults['runs_dir'],
                       help="Path to TensorBoard runs directory.")
    
    # Colab optimization
    parser.add_argument('--no-local-copy', action='store_true',
                       help="Disable copying data to local storage in Colab (slower but saves space)")
    
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
    
    # Logging metrics to .csv option 
    parser.add_argument('--metric', action='store_true',
                        help="Enable storing results in .csv after each epoch")
    
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
        runs_dir=args.runs_dir,
        metric=args.metric, 
        use_local_copy=not args.no_local_copy,
        optimizer_type=args.optimizer,
        momentum=args.momentum, 
        **hyperparams
    )