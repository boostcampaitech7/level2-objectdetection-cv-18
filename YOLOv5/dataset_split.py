import os
import random
import shutil
import stat
from text import process_labels  # Import the function from text.py

# Set the seed for reproducibility
random.seed(42)

# Define the source dataset folder and target train/val folders
dataset_folder = '/data/ephemeral/home/dataset_origin/train'  # Replace with your actual dataset folder path
train_folder = '/data/ephemeral/home/jiwan/dataset/train/images'
val_folder = '/data/ephemeral/home/jiwan/dataset/val/images'
train_labels_dir = '/data/ephemeral/home/jiwan/dataset/train/labels'
val_labels_dir = '/data/ephemeral/home/jiwan/dataset/val/labels'

# JSON file path
json_file_path = '/data/ephemeral/home/dataset_origin/train.json'  # Replace with your actual JSON file path

# Image size configuration (adjust if needed)
img_width = 1024
img_height = 1024

# Helper function to forcefully delete read-only files
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Function to delete and recreate a folder
def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, onerror=remove_readonly)  # Force deletion
    os.makedirs(folder_path)  # Recreate the folder

# Reset train/val directories
reset_folder(train_folder)
reset_folder(val_folder)
reset_folder(train_labels_dir)
reset_folder(val_labels_dir)

# Get list of all images in the dataset folder
images = [img for img in os.listdir(dataset_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the images
random.shuffle(images)

# Calculate split point for 80% train and 20% val
split_point = int(len(images) * 0.8)

# Split images into train and val sets
train_images = images[:split_point]
val_images = images[split_point:]

# Copy images to the respective folders
for img in train_images:
    shutil.copy(os.path.join(dataset_folder, img), os.path.join(train_folder, img))

for img in val_images:
    shutil.copy(os.path.join(dataset_folder, img), os.path.join(val_folder, img))

print(f"Copied {len(train_images)} images to {train_folder} and {len(val_images)} images to {val_folder}.")

# Call the function from text.py to process and save the labels
process_labels(json_file_path, img_width, img_height, train_images, val_images, train_labels_dir, val_labels_dir)
