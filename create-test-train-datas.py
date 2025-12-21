import os
import shutil
import random

# Folder paths
BASE_DIR = "data"
SOURCE_DIR = os.path.join(BASE_DIR, "animals")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# 80% train 20% test 
SPLIT_RATIO = 0.8

# Create train and test folders if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Create and copy images to train/test subfolders
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # Create train and test sub folders
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Take images
    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy the images to train
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy(src, dst)

    # Copy the images to test
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy(src, dst)

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("Successful")

