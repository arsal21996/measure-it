
import cv2
import numpy as np
from pathlib import Path

# Calibration settings
IMAGE_DIR = Path("calibration/images")
OUTPUT_DIR = Path("calibration")
PATTERN_SIZE = (8, 6)  # Inner corners: columns, rows
SQUARE_SIZE = 25.0     # Millimetres

# Prepare checkerboard 3D points
objp = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    np.float32
)
objp[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0],
    0:PATTERN_SIZE[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

object_points = []
image_points = []
image_size = None
successful_images = []
failed_images = []

# Find image files
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
)

print(f"Found {len(image_paths)} images.")

# Detect checkerboard corners
for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        failed_images.append(image_path.name)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    # Try the newer detector first
    found, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Fall back to the standard detector
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if found:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        object_points.append(objp.copy())
        image_points.append(refined)

        successful_images.append(image_path.name)
        print(f"Detected: {image_path.name}")

    else:
        failed_images.append(image_path.name)
        print(f"Not detected: {image_path.name}")

# Detection summary
print("\n--- Detection summary ---")
print(f"Successful: {len(successful_images)}")
print(f"Failed: {len(failed_images)}")

if len(successful_images) < 10:
    raise RuntimeError(
        "Fewer than 10 usable images. Calibration stopped."
    )

# Calculate camera calibration
rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

# Calculate mean reprojection error
total_error = 0.0

for i in range(len(object_points)):

    projected, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(
        image_points[i],
        projected,
        cv2.NORM_L2
    ) / np.sqrt(len(projected))

    total_error += error
    print(f"Reprojection error - {successful_images[i]}: {error:.3f} pixels")

mean_error = total_error / len(object_points)

# Save calibration results
np.savez(
    OUTPUT_DIR / "camera_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rms_error=rms,
    mean_reprojection_error=mean_error,
    image_width=image_size[0],
    image_height=image_size[1]
)

# Display results
print("\n--- Calibration results ---")
print("Camera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)

print(f"\nRMS error: {rms:.4f}")
print(f"Mean reprojection error: {mean_error:.4f} pixels")

print(
    f"\nSaved to: "
    f"{OUTPUT_DIR / 'camera_calibration.npz'}"
)
