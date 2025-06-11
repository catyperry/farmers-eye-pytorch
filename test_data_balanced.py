# Create test data set - Balanced set
# 85 images per class!

import os
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# the csv contains 85 images per class and creates a balanced test set

# --- Define paths
base_drive_path = '.'
source_image_dir = os.path.join(base_drive_path, 'full_data_set')  # e.g., images/B11/*.jpg
csv_path = os.path.join(base_drive_path, 'full_data_set/lucasvision_MMEC_testSet85.csv')  # update this
temp_output_dir = os.path.join(base_drive_path, 'inputs/test_balanced')

os.makedirs(temp_output_dir, exist_ok=True)

# --- Load CSV
df = pd.read_csv(csv_path)

# --- Clean and get only necessary columns
# df = df[df['trainok'] == True]
df = df[['lc1', 'filepath_ftp']]  # lc1 is the label (B11, B12), filepath_ftp contains the name

# --- Convert FTP filepath to actual filename
df['filename'] = df['filepath_ftp'].apply(lambda x: os.path.basename(str(x)))

success_count = defaultdict(int)
missing_count = defaultdict(int)

# --- Loop through each row and copy image
for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
    label = row['lc1'].strip()
    filename = row['filename'].strip()
    
    src = os.path.join(source_image_dir, label, filename)
    dst_folder = os.path.join(temp_output_dir, label)
    dst = os.path.join(dst_folder, filename)
    
    os.makedirs(dst_folder, exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        success_count[label] += 1
    else:
        missing_count[label] += 1

# --- Print summary
print("\nSummary per class:")
for label in sorted(set(df['lc1'])):
    print(f"{label}: Copied {success_count[label]}, Missing {missing_count[label]}")