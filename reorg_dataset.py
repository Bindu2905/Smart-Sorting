import os
import shutil
import random

# Path where your downloaded dataset exists
RAW_DIR = "raw_dataset"
DEST_DIR = "dataset"
TRAIN_RATIO = 0.8  # 80% train, 20% test

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# Find all subfolders inside fruits/ and vegetables/
categories = []
for main_folder in os.listdir(RAW_DIR):
    main_path = os.path.join(RAW_DIR, main_folder)
    if not os.path.isdir(main_path):
        continue
    for category in os.listdir(main_path):
        cat_path = os.path.join(main_path, category)
        if os.path.isdir(cat_path):
            categories.append(cat_path)

print("Found categories:", [os.path.basename(c) for c in categories])

# Split images for each category
for cat_path in categories:
    images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    split_index = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    cat_name = os.path.basename(cat_path)

    for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
        dest_folder = os.path.join(DEST_DIR, split, cat_name)
        os.makedirs(dest_folder, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(cat_path, img), os.path.join(dest_folder, img))

print("✅ Dataset organized into:")
print("   dataset/train/... and dataset/test/...")