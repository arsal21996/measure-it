
import cv2
import numpy as np

calibration = np.load("calibration/camera_calibration.npz")
camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

image = cv2.imread("calibration/images/20261119_154304.jpg")

if image is None:
    raise FileNotFoundError("Test image not found.")

# Resize to the exact calibration resolution
image = cv2.resize(image, (1200, 1600))

undistorted = cv2.undistort(
    image,
    camera_matrix,
    dist_coeffs
)

cv2.imwrite("calibration/undistorted_1200x1600.jpg", undistorted)

print("Saved calibration-resolution test image.")
