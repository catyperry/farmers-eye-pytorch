# Create test data set - Balanced set
# 85 images per class!

import os
import shutil
import pandas as pd

# --- Define paths
base_drive_path = '/content/drive/My Drive/farmers_eye'
source_image_dir = os.path.join(base_drive_path, 'full_data_set')  # e.g., images/B11/*.jpg
csv_path = os.path.join(base_drive_path, 'full_data_set/lucasvision_MMEC_testSet85.csv')  # update this
temp_output_dir = os.path.join(base_drive_path, 'inputs/test85')

# --- Load CSV
df = pd.read_csv(csv_path)

# --- Clean and get only necessary columns
df = df[df['trainok'] == True]
df = df[['lc1', 'filepath_ftp']]  # lc1 is the label (B11, B12), filepath_ftp contains the name

# --- Convert FTP filepath to actual filename
df['filename'] = df['filepath_ftp'].apply(lambda x: os.path.basename(str(x)))

# --- Loop through each row and copy image
for _, row in df.iterrows():
    label = row['lc1']
    filename = row['filename']
    
    src = os.path.join(source_image_dir, label, filename)
    dst_folder = os.path.join(temp_output_dir, label)
    dst = os.path.join(dst_folder, filename)
    
    os.makedirs(dst_folder, exist_ok=True)
    
    try:
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    except FileNotFoundError:
        print(f"WARNING: File not found: {src}")