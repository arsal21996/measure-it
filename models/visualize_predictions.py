
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

IMAGE_DIR = Path("dataset/images/test")
MASK_DIR = Path("dataset/masks/test")
CHECKPOINT = Path("models/best_segmentation_model.pth")
OUTPUT_DIR = Path("models/prediction_visualizations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Build model architecture
print("Loading trained model...")

model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    aux_loss=True
)

model.classifier = DeepLabHead(2048, 2)

model.aux_classifier = torch.nn.Sequential(
    torch.nn.Conv2d(1024, 256, 3, padding=1, bias=False),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Conv2d(256, 2, 1)
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# Process test images
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError("No test images found.")

print(f"Generating visualizations for {len(image_paths)} images...")


for image_path in image_paths:

    mask_path = MASK_DIR / (image_path.stem + ".png")

    if not mask_path.exists():
        print(f"Skipping missing mask: {image_path.name}")
        continue

    # Load original image and ground truth
    original = Image.open(image_path).convert("RGB")

    ground_truth = Image.open(mask_path).convert("L")

    # Resize consistently with training
    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    ground_truth = ground_truth.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    image_array = np.asarray(
        resized, dtype=np.float32
    ) / 255.0

    mask_array = np.asarray(
        ground_truth, dtype=np.uint8
    )

    # Normalize image
    normalized = (image_array - MEAN) / STD

    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0).to(DEVICE)

    # Predict segmentation
    with torch.no_grad():
        output = model(tensor)["out"]
        prediction = output.argmax(dim=1)[0].cpu().numpy()

    ground_truth_binary = mask_array > 127

    # Calculate individual IoU
    intersection = np.logical_and(
        prediction == 1,
        ground_truth_binary
    ).sum()

    union = np.logical_or(
        prediction == 1,
        ground_truth_binary
    ).sum()

    iou = intersection / max(union, 1)

    # Create prediction overlay
    overlay = image_array.copy()

    # Highlight predicted object in green
    predicted_pixels = prediction == 1

    overlay[predicted_pixels] = (
        0.35 * overlay[predicted_pixels]
        + 0.65 * np.array([0, 1, 0])
    )

    # Save comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")

    axes[1].imshow(ground_truth_binary, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Predicted Overlay | IoU: {iou:.4f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_path.name)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path.name} | IoU: {iou:.4f}")


print("\nVisualization completed.")
print(f"Results saved in: {OUTPUT_DIR}")
