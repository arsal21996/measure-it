
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# ---------------- CONFIGURATION ----------------

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- DATASET ----------------

class GlassesDataset(Dataset):
    def __init__(self, split):
        self.image_dir = BASE_DIR / "images" / split
        self.mask_dir = BASE_DIR / "masks" / split

        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

        for image_path in self.images:
            mask_path = self.mask_dir / (image_path.stem + ".png")
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / (image_path.stem + ".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.uint8)

        # Convert mask to class IDs: 0=background, 1=object
        mask = (mask > 127).astype(np.int64)

        # Random horizontal flip for training only
        if self.training and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image = (image - MEAN) / STD

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()

        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

    @property
    def training(self):
        return self.image_dir.name == "train"


# ---------------- LOAD DATA ----------------

train_dataset = GlassesDataset("train")
val_dataset = GlassesDataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Device: {DEVICE}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ---------------- MODEL ----------------

print("\nLoading pretrained DeepLabV3 model...")

model = deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.DEFAULT
)

# Replace segmentation head for two classes
model.classifier = DeepLabHead(2048, 2)

# Replace auxiliary classifier
if model.aux_classifier is not None:
    model.aux_classifier = nn.Sequential(
        nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, 2, kernel_size=1)
    )

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ---------------- METRICS ----------------

def calculate_metrics(predictions, targets):
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    intersection = ((predictions == 1) & (targets == 1)).sum().item()
    union = ((predictions == 1) | (targets == 1)).sum().item()

    true_positive = intersection
    false_positive = (
        (predictions == 1) & (targets == 0)
    ).sum().item()

    false_negative = (
        (predictions == 0) & (targets == 1)
    ).sum().item()

    iou = intersection / max(union, 1)
    precision = true_positive / max(
        true_positive + false_positive, 1
    )
    recall = true_positive / max(
        true_positive + false_negative, 1
    )

    return iou, precision, recall


# ---------------- VALIDATION ----------------

def validate():
    model.eval()

    total_loss = 0.0
    total_intersection = 0
    total_union = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            predictions = outputs.argmax(dim=1)

            pred_object = predictions == 1
            true_object = masks == 1

            total_intersection += (
                pred_object & true_object
            ).sum().item()

            total_union += (
                pred_object | true_object
            ).sum().item()

            total_tp += (pred_object & true_object).sum().item()
            total_fp += (pred_object & ~true_object).sum().item()
            total_fn += (~pred_object & true_object).sum().item()

            total_loss += loss.item()

    iou = total_intersection / max(total_union, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    return (
        total_loss / len(val_loader),
        iou,
        precision,
        recall
    )


# ---------------- TRAINING ----------------

best_iou = -1.0
history = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for batch_index, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        main_loss = criterion(outputs["out"], masks)

        if "aux" in outputs:
            aux_loss = criterion(outputs["aux"], masks)
            loss = main_loss + 0.4 * aux_loss
        else:
            loss = main_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_loss, val_iou, precision, recall = validate()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"IoU: {val_iou:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )

    history.append([
        epoch + 1,
        train_loss,
        val_loss,
        val_iou,
        precision,
        recall
    ])

    if val_iou > best_iou:
        best_iou = val_iou

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "classes": ["background", "glasses_case"],
                "best_val_iou": best_iou,
            },
            MODEL_DIR / "best_segmentation_model.pth"
        )

        print("  Best model checkpoint saved.")

# Save training history
with open(
    MODEL_DIR / "training_history.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "train_loss", "val_loss",
        "val_iou", "precision", "recall"
    ])
    writer.writerows(history)

print("\nTraining completed.")
print(f"Best validation IoU: {best_iou:.4f}")
print("Model: models/best_segmentation_model.pth")
print("History: models/training_history.csv")
