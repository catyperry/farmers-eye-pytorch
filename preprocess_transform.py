import os
import sys
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import argparse

def is_colab():
    """
    Check if running in Google Colab.
    This is more robust than checking sys.modules, as it works in shell subprocesses.
    """
    return 'COLAB_GPU' in os.environ

def get_default_paths_for_preprocessing():
    """Get default paths for the preprocessing script based on environment."""
    if is_colab():
        # In Colab, we read from a 'raw_images' folder on Drive and write
        # to the 'data' folder that the training script will use as its source.
        return {
            'train_data': '/content/drive/MyDrive/farmers_eye/inputs/training',
            'test_balanced_data': '/content/drive/MyDrive/farmers_eye/inputs/test_balanced',
            'processed_train_data': '/content/data_train_local',
            'processed_test_balanced_data': '/content/data_test_local',
        }
    else:
        # For local development, we use local relative paths.
        return {
            'train_data': './inputs/training',   ####### --- Changed to "training" @silke -----#######
            'test_balanced_data': './inputs/test_balanced',
            'processed_train_data': './inputs/data_train_local',
            'processed_test_balanced_data': './inputs/data_test_local',
        }

def preprocess_dataset(data_dir, output_dir):
    """
    Preprocesses a directory of images and saves them as tensors.
    The class_to_idx.pt file is saved inside the output_dir.
    """
    print(f"Processing images from: {data_dir}")
    print(f"Saving preprocessed tensors to: {output_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Input data directory not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    dataset = ImageFolder(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create a subdirectory for each class within the output directory
    for class_name in dataset.classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Process and save each image as a tensor
    for path, label in tqdm(dataset.samples, desc="Preprocessing"):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            class_name = dataset.classes[label]
            filename = os.path.splitext(os.path.basename(path))[0] + ".pt"
            save_path = os.path.join(output_dir, class_name, filename)
            torch.save((img_tensor, label), save_path)
        except Exception as e:
            print(f"\nSkipping file {path} due to error: {e}")

    # Save the class index map *inside* the main output directory
    class_map_path = os.path.join(output_dir, 'class_to_idx.pt')
    torch.save(dataset.class_to_idx, class_map_path)
    print(f"Class index map saved to: {class_map_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    defaults = get_default_paths_for_preprocessing()
    
    parser = argparse.ArgumentParser(description="Preprocess image datasets into tensors.")
    
    # A single, required argument to choose the dataset
    parser.add_argument(
        "--dataset", 
        required=True, 
        choices=['train', 'test_balanced'],
        help="The name of the dataset to process (e.g., 'train' or 'test_balanced')."
    )
    args = parser.parse_args()
    
    # Automatically construct the input and output paths based on the choice
    data_dir = defaults['train_data']
    output_dir = defaults['processed_train_data']
    if args.dataset == 'test_balanced':
        data_dir = defaults['test_balanced_data']
        output_dir = defaults['processed_test_balanced_data']

    # Run the preprocessing function
    preprocess_dataset(data_dir, output_dir)