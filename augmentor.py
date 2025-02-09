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

# Define augmentation functions with reasonable parameters
def augment_1(img):
    """Moderately increase saturation"""
    return imaugs.Saturation(factor=1.3)(img)

def augment_2(img):
    """Add a small emoji overlay with partial transparency"""
    return imaugs.OverlayEmoji(opacity=0.3, emoji_size=0.1)(img)

def augment_3(img):
    """Pad image to square while maintaining aspect ratio"""
    return imaugs.PadSquare()(img)

def augment_4(img):
    """Apply mild perspective transform"""
    return imaugs.PerspectiveTransform(sigma=10.0)(img)

def augment_5(img):
    """Slightly adjust brightness and contrast"""
    return imaugs.ColorJitter(brightness_factor=1.1, contrast_factor=1.2)(img)

def augment_6(img):
    """Minimally shuffle pixels to maintain image recognition"""
    return imaugs.ShufflePixels(factor=0.1)(img)

def augment_7(img):
    """Modify aspect ratio within reasonable bounds"""
    return imaugs.RandomAspectRatio(min_ratio=0.9, max_ratio=1.1)(img)

def augment_8(img):
    """Moderately increase brightness"""
    return imaugs.Brightness(factor=1.2)(img)

def augment_9(img):
    """Add subtle stripe overlay"""
    return imaugs.OverlayStripes(line_width=0.1, line_opacity=0.3, line_angle=45.0)(img)

def augment_10(img):
    """Moderately adjust contrast"""
    return imaugs.Contrast(factor=1.3)(img)

# List of augmentation functions
augment_functions = [augment_1, augment_2, augment_3, augment_4, augment_5,
                     augment_6, augment_7, augment_8, augment_9, augment_10]

# Augmented data storage
augmented_images = []
augmented_labels = []

# Apply augmentations: Each image gets 2 randomly augmented versions
for i, img in enumerate(train_images_pil):
    # Store original image
    augmented_images.append(np.array(img))  
    augmented_labels.append(train_labels[i])

    # Create two augmented versions
    for _ in range(2):
        aug_img = img.copy()  # Start from original
        selected_transforms = random.sample(augment_functions, 3)  # Pick 3 unique transforms

        # Apply transformations sequentially
        for func in selected_transforms:
            aug_img = func(aug_img)

        # Convert back to NumPy (Ensuring dtype consistency)
        augmented_images.append(np.array(aug_img, dtype=np.uint8))
        augmented_labels.append(train_labels[i])

# Resize images to match ResNet50 input (224x224)
IMG_SHAPE = (224, 224)
augmented_images = [Image.fromarray(img).resize(IMG_SHAPE) for img in augmented_images]

# Convert back to NumPy arrays
augmented_images = np.array([np.array(img, dtype=np.uint8) for img in augmented_images])
augmented_labels = np.array(augmented_labels)

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

print("Augmentation complete. Final dataset saved in 'augmented_train/'")