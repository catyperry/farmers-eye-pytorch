# preprocess_images.py
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse

def preprocess_dataset(data_dir, output_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    dataset = ImageFolder(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(dataset.classes):
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    for idx, (path, label) in enumerate(tqdm(dataset.samples, desc="Preprocessing")):
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        class_name = dataset.classes[label]
        filename = os.path.splitext(os.path.basename(path))[0] + ".pt"
        save_path = os.path.join(output_dir, class_name, filename)
        torch.save((img_tensor, label), save_path)

    torch.save(dataset.class_to_idx, os.path.join(output_dir, 'class_to_idx.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to original images.")
    parser.add_argument("--output_dir", required=True, help="Where to save preprocessed tensors.")
    args = parser.parse_args()
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    preprocess_dataset(args.data_dir, args.output_dir)
