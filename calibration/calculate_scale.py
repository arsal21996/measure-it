
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math
import statistics
import csv

IMAGE_DIR = Path("calibration/scale_images")
OUTPUT_CSV = Path("calibration/scale_results.csv")

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if len(images) != 5:
    raise RuntimeError(f"Expected 5 ruler photos, found {len(images)}")

results = []

print("\nRULER SCALE CALIBRATION")
print("For each image:")
print("1. Click the center of one clear centimetre marking.")
print("2. Click another centimetre marking.")
print("3. Enter the distance between those markings in cm.")
print("Use widely separated markings for better accuracy.\n")

for index, path in enumerate(images, 1):
    img = Image.open(path).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(
        f"Image {index}/5: {path.name}\n"
        "Click two ruler marks, then close the image window."
    )
    ax.axis("on")
    plt.tight_layout()
    plt.show(block=False)

    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(points) != 2:
        raise RuntimeError("Two points were not selected.")

    distance_cm = float(input(
        f"{path.name} - distance between selected marks (cm): "
    ))

    if distance_cm <= 0:
        raise ValueError("Distance must be positive.")

    pixel_distance = math.dist(points[0], points[1])
    scale = pixel_distance / (distance_cm * 10)

    results.append({
        "image": path.name,
        "pixel_distance": pixel_distance,
        "known_distance_mm": distance_cm * 10,
        "pixels_per_mm": scale,
        "mm_per_pixel": 1 / scale
    })

    print(f"Scale: {scale:.4f} pixels/mm\n")

mean_scale = statistics.mean(r["pixels_per_mm"] for r in results)
std_scale = statistics.stdev(r["pixels_per_mm"] for r in results)
variation = std_scale / mean_scale * 100

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== CALIBRATION RESULTS ==========")
print(f"Mean scale: {mean_scale:.4f} pixels/mm")
print(f"Mean mm/pixel: {1 / mean_scale:.6f}")
print(f"Variation: {variation:.2f}%")
print(f"Saved results: {OUTPUT_CSV}")
