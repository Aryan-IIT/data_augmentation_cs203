import augly.image as imaugs
import numpy as np
import os
import random
from PIL import Image

# Load dataset (Ensure RGB)
train_images = np.load("train_dataset/train_images.npy").astype(np.uint8)  # Ensure uint8 for proper PIL conversion
train_labels = np.load("train_dataset/train_labels.npy")

# Ensure output folder exists
os.makedirs("augmented_train", exist_ok=True)

# Convert NumPy arrays to PIL images for augmentation
train_images_pil = [Image.fromarray(img) for img in train_images]  # No need for *255 since it's already uint8

# Define augmentation functions (Ensuring RGB retention)
def augment_1(img): return imaugs.Saturation(factor=2.0)(img)
def augment_2(img): return imaugs.OverlayEmoji(opacity=0.7, emoji_size=0.2)(img)
def augment_3(img): return imaugs.PadSquare()(img)
def augment_4(img): return imaugs.PerspectiveTransform(sigma=25.0)(img)
def augment_5(img): return imaugs.ColorJitter(brightness_factor=1.3, contrast_factor=1.5)(img)
def augment_6(img): return imaugs.ShufflePixels(factor=0.3)(img)
def augment_7(img): return imaugs.RandomAspectRatio()(img)
def augment_8(img): return imaugs.Brightness(factor=1.5)(img)
def augment_9(img): return imaugs.OverlayStripes(line_opacity=0.6, line_angle=30.0)(img)
def augment_10(img): return imaugs.Contrast(factor=1.8)(img)

# List of augmentation functions
augment_functions = [augment_1, augment_2, augment_3, augment_4, augment_5,
                     augment_6, augment_7, augment_8, augment_9, augment_10]

# Augmented data storage
augmented_images = []
augmented_labels = []

# Apply augmentations: Forward pass (1 → 10)
for i in range(112):
    func = augment_functions[i % 10]
    aug_img = func(train_images_pil[i])
    augmented_images.append(np.array(aug_img))  # Ensure NumPy conversion
    augmented_labels.append(train_labels[i])

# Apply augmentations: Backward pass (10 → 1)
for i in range(112):
    func = augment_functions[9 - (i % 10)]
    aug_img = func(train_images_pil[i])
    augmented_images.append(np.array(aug_img))
    augmented_labels.append(train_labels[i])

# Add original images
augmented_images.extend(train_images)  # Already NumPy arrays, no need to convert
augmented_labels.extend(train_labels)

# Resize images to match ResNet50 input (224x224)
IMG_SHAPE = (224, 224)
augmented_images = [Image.fromarray(img).resize(IMG_SHAPE) for img in augmented_images]

# Convert back to NumPy (Ensuring dtype consistency)
augmented_images = np.array([np.array(img, dtype=np.uint8) for img in augmented_images])

# Shuffle dataset
seed = 12
combined = list(zip(augmented_images, augmented_labels))
random.seed(seed)
random.shuffle(combined)
augmented_images, augmented_labels = zip(*combined)

# Convert back to NumPy arrays
augmented_images = np.array(augmented_images, dtype=np.uint8)
augmented_labels = np.array(augmented_labels)

# Save dataset
np.save("augmented_train/train_images.npy", augmented_images)
np.save("augmented_train/train_labels.npy", augmented_labels)

print("✅ Augmentation complete. Final dataset saved in 'augmented_train/'")