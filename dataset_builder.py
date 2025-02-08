import os
import shutil
import random
import numpy as np
import cv2

# Paths
src_dir = "test"
train_dir = "train_dataset"
test_dir = "test_dataset"
split_ratio = 0.8  # 80% train, 20% test

# Ensure output directories exist
for folder in [train_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Data storage
train_data, train_labels = [], []
test_data, test_labels = [], []

# Assign numeric labels (e.g., cat=0, dog=1)
label_map = {"cats": 0, "dogs": 1}

# Process each category
for category in os.listdir(src_dir):
    cat_path = os.path.join(src_dir, category)
    if not os.path.isdir(cat_path):
        continue  # Skip non-folder files

    images = [img for img in os.listdir(cat_path) if img.endswith(".jpg")]
    random.shuffle(images)  # Shuffle for randomness

    split_index = int(len(images) * split_ratio)
    train_images, test_images = images[:split_index], images[split_index:]

    # Create category folders in train and test directories
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Process train images
    for img in train_images:
        img_path = os.path.join(cat_path, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_array = cv2.resize(img_array, (224, 224))

        train_data.append(img_array)
        train_labels.append(label_map[category])

        shutil.copy(img_path, os.path.join(train_dir, category, img))  # Copy image

    # Process test images
    for img in test_images:
        img_path = os.path.join(cat_path, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_array = cv2.resize(img_array, (224, 224))

        test_data.append(img_array)
        test_labels.append(label_map[category])

        shutil.copy(img_path, os.path.join(test_dir, category, img))

# Save as NumPy arrays
np.save(os.path.join(train_dir, "train_images.npy"), np.array(train_data))
np.save(os.path.join(train_dir, "train_labels.npy"), np.array(train_labels))
np.save(os.path.join(test_dir, "test_images.npy"), np.array(test_data))
np.save(os.path.join(test_dir, "test_labels.npy"), np.array(test_labels))

print("Train-test split completed, and image arrays saved! 🚀")