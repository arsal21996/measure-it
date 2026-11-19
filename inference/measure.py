
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_segmentation_model.pth"
CALIBRATION_PATH = ROOT / "calibration" / "camera_calibration.npz"
OUTPUT_DIR = ROOT / "inference" / "results"

# Provisional scale: median from five ruler photographs.
# The observed variation was 10.71%, so measurements are estimates.
PIXELS_PER_MM = 4.2175

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------- MODEL ----------------

def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=True
    )

    model.classifier = DeepLabHead(2048, 2)

    model.aux_classifier = nn.Sequential(
        nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, 2, kernel_size=1)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, checkpoint["image_size"]


# ---------------- IMAGE PROCESSING ----------------

def load_and_undistort(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    calibration = np.load(CALIBRATION_PATH)

    width = int(calibration["image_width"])
    height = int(calibration["image_height"])

    # Calibration images were portrait (1200 x 1600).
    # Rotate landscape input into that orientation first.
    rotate_back = image.shape[1] > image.shape[0]

    if rotate_back:
        image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    original_ratio = image.shape[1] / image.shape[0]
    calibration_ratio = width / height

    if abs(original_ratio - calibration_ratio) > 0.01:
        raise ValueError(
            "Image aspect ratio differs from calibration."
        )

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    undistorted = cv2.undistort(
        image,
        calibration["camera_matrix"],
        calibration["dist_coeffs"]
    )

    # Restore original landscape orientation.
    if rotate_back:
        undistorted = cv2.rotate(
            undistorted,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return undistorted


# ---------------- SEGMENTATION ----------------

def predict_mask(model, image, input_size):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD

    tensor = torch.from_numpy(
        tensor.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(tensor)["out"]
        probabilities = torch.softmax(output, dim=1)

    foreground_probability = (
        probabilities[0, 1].cpu().numpy()
    )

    small_mask = np.argmax(
        probabilities[0].cpu().numpy(),
        axis=0
    ).astype(np.uint8)

    height, width = image.shape[:2]

    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    foreground_probability = cv2.resize(
        foreground_probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return mask, foreground_probability


# ---------------- MEASUREMENT ----------------

def measure_object(mask, foreground_probability):
    binary = (mask == 1).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(binary, 8)
    )

    if count <= 1:
        raise RuntimeError(
            "Glasses case not detected. No measurement produced."
        )

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    largest_mask = (labels == largest_label).astype(np.uint8)

    contours, _ = cv2.findContours(
        largest_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No valid object contour found.")

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 500:
        raise RuntimeError("Detected object is too small.")

    rectangle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rectangle)
    box = np.int32(box)

    dimension_a, dimension_b = rectangle[1]

    length_px = max(dimension_a, dimension_b)
    width_px = min(dimension_a, dimension_b)

    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM

    selected_pixels = foreground_probability[largest_mask == 1]

    confidence = (
        float(np.mean(selected_pixels))
        if selected_pixels.size else 0.0
    )

    return {
        "mask": largest_mask,
        "box": box,
        "length_px": float(length_px),
        "width_px": float(width_px),
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "confidence": confidence,
        "area_px": int(cv2.contourArea(contour))
    }


# ---------------- OUTPUT ----------------

def save_results(image, measurement, image_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = image_path.stem

    mask = measurement["mask"] * 255

    overlay = image.copy()

    foreground = measurement["mask"] == 1

    overlay[foreground] = (
        0.4 * overlay[foreground]
        + 0.6 * np.array([0, 255, 0])
    ).astype(np.uint8)

    cv2.drawContours(
        overlay,
        [measurement["box"]],
        -1,
        (0, 0, 255),
        3
    )

    mask_path = OUTPUT_DIR / f"{name}_mask.png"
    overlay_path = OUTPUT_DIR / f"{name}_overlay.jpg"
    report_path = OUTPUT_DIR / f"{name}_measurement.json"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    report = {
        "image": image_path.name,
        "length_pixels": round(measurement["length_px"], 2),
        "width_pixels": round(measurement["width_px"], 2),
        "length_mm_provisional": round(measurement["length_mm"], 2),
        "width_mm_provisional": round(measurement["width_mm"], 2),
        "scale_pixels_per_mm": PIXELS_PER_MM,
        "mean_foreground_probability": round(
            measurement["confidence"], 4
        ),
        "measurement_status": "PROVISIONAL",
        "scale_note": (
            "Scale variation across five ruler images was 10.71%. "
            "Millimetre values are estimates, not validated measurements."
        )
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment and measure a glasses case."
    )

    parser.add_argument(
        "image",
        help="Path to input photograph"
    )

    args = parser.parse_args()
    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    print("Device:", DEVICE)
    print("Loading trained segmentation model...")

    model, input_size = load_model()

    print("Loading and undistorting image...")
    image = load_and_undistort(image_path)

    print("Segmenting glasses case...")
    mask, probabilities = predict_mask(
        model,
        image,
        input_size
    )

    print("Calculating dimensions...")
    measurement = measure_object(mask, probabilities)

    report = save_results(
        image,
        measurement,
        image_path
    )

    print("\n========== MEASUREMENT REPORT ==========")
    print("Image:", report["image"])
    print(f"Length: {report['length_pixels']:.2f} pixels")
    print(f"Width:  {report['width_pixels']:.2f} pixels")
    print(
        f"Length: {report['length_mm_provisional']:.2f} mm "
        "(provisional)"
    )
    print(
        f"Width:  {report['width_mm_provisional']:.2f} mm "
        "(provisional)"
    )
    print(
        "Foreground probability:",
        report["mean_foreground_probability"]
    )
    print("Status:", report["measurement_status"])
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
