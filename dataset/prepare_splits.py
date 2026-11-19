
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("dataset/images")
MANIFEST_DIR = Path("dataset/splits")
SEED = 42

# Find original images directly inside dataset/images
images = sorted(
    p for p in SOURCE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

total = len(images)

if total < 70:
    raise ValueError(f"At least 70 images are required. Found {total}.")

# Shuffle reproducibly
random.Random(SEED).shuffle(images)

# Calculate 70/20/10 split
train_count = int(total * 0.70)
val_count = int(total * 0.20)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images,
}

# Copy images into their respective folders
for split_name, split_images in splits.items():
    destination = SOURCE_DIR / split_name
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in split_images:
        shutil.copy2(image_path, destination / image_path.name)

# Save reproducible split manifest
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

with open(MANIFEST_DIR / "split_manifest.txt", "w", encoding="utf-8") as f:
    for split_name, split_images in splits.items():
        for image_path in split_images:
            f.write(f"{split_name}/{image_path.name}\n")

# Report results
print("Dataset split completed successfully.")
print(f"Total images: {total}")

for split_name, split_images in splits.items():
    percentage = len(split_images) / total * 100
    print(f"{split_name}: {len(split_images)} images ({percentage:.1f}%)")

print(f"Manifest saved to: {MANIFEST_DIR / 'split_manifest.txt'}")
