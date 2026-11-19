
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# ================= CONFIGURATION =================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

IMAGE_DIR = Path("dataset/images/test")
MASK_DIR = Path("dataset/masks/test")
CHECKPOINT = Path("models/best_segmentation_model.pth")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ================= TEST DATASET =================

class TestDataset(Dataset):

    def __init__(self):
        self.images = sorted(
            p for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not self.images:
            raise RuntimeError("No test images found.")

        for path in self.images:
            mask_path = MASK_DIR / (path.stem + ".png")

            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        path = self.images[index]
        mask_path = MASK_DIR / (path.stem + ".png")

        image = Image.open(path).convert("RGB")
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
        image = (image - MEAN) / STD

        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask > 127).astype(np.int64)

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()

        mask = torch.from_numpy(mask.copy()).long()

        return image, mask, path.name


# ================= LOAD MODEL =================

print("Loading model architecture...")

model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    aux_loss=True
)

# Main classifier: background and glasses case
model.classifier = DeepLabHead(2048, 2)

# Auxiliary classifier: same structure as training
model.aux_classifier = torch.nn.Sequential(
    torch.nn.Conv2d(
        1024, 256,
        kernel_size=3,
        padding=1,
        bias=False
    ),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Conv2d(256, 2, kernel_size=1)
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")
print(f"Device: {DEVICE}")
print(f"Best validation IoU: {checkpoint['best_val_iou']:.4f}")


# ================= LOAD TEST DATA =================

dataset = TestDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"\nEvaluating {len(dataset)} test images...\n")


# ================= EVALUATION =================

total_intersection = 0
total_union = 0
total_tp = 0
total_fp = 0
total_fn = 0

with torch.no_grad():

    for images, masks, names in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)["out"]
        predictions = outputs.argmax(dim=1)

        pred_object = predictions == 1
        true_object = masks == 1

        intersection = (
            pred_object & true_object
        ).sum().item()

        union = (
            pred_object | true_object
        ).sum().item()

        tp = intersection

        fp = (
            pred_object & ~true_object
        ).sum().item()

        fn = (
            ~pred_object & true_object
        ).sum().item()

        total_intersection += intersection
        total_union += union
        total_tp += tp
        total_fp += fp
        total_fn += fn

        image_iou = intersection / max(union, 1)

        print(f"{names[0]} | IoU: {image_iou:.4f}")


# ================= FINAL METRICS =================

iou = total_intersection / max(total_union, 1)

precision = total_tp / max(
    total_tp + total_fp, 1
)

recall = total_tp / max(
    total_tp + total_fn, 1
)

print("\n========== TEST RESULTS ==========")
print(f"Test images: {len(dataset)}")
print(f"IoU:         {iou:.4f} ({iou * 100:.2f}%)")
print(f"Precision:   {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:      {recall:.4f} ({recall * 100:.2f}%)")
print("==================================")
