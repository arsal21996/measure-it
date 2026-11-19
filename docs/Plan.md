\# Project Plan — MEASURE IT



\## Project Overview

Build a computer-vision system that segments a glasses case in an image and estimates its physical length and width in millimeters.



\## Implementation Plan



1\. \*\*Environment Setup\*\*

&#x20;  - Create an isolated Python virtual environment.

&#x20;  - Install and verify pinned dependencies.

&#x20;  - Confirm NVIDIA GPU acceleration.



2\. \*\*Camera Calibration\*\*

&#x20;  - Capture at least 20 checkerboard images.

&#x20;  - Calculate and save the camera matrix, distortion coefficients, and reprojection error.

&#x20;  - Implement image undistortion.



3\. \*\*Dataset Preparation\*\*

&#x20;  - Collect 84 images of the glasses case.

&#x20;  - Create binary segmentation masks.

&#x20;  - Split the dataset into training, validation, and test sets.

&#x20;  - Save the split manifest for reproducibility.



4\. \*\*Segmentation Model\*\*

&#x20;  - Train a DeepLabV3-ResNet50 model using the prepared dataset.

&#x20;  - Save the best validation checkpoint and training history.

&#x20;  - Evaluate the model on the held-out test set.



5\. \*\*Measurement Pipeline\*\*

&#x20;  - Load and undistort an input image.

&#x20;  - Predict the glasses-case segmentation mask.

&#x20;  - Extract the object's boundary and estimate its pixel dimensions.

&#x20;  - Convert pixel dimensions to millimeters using a reference-derived scale.

&#x20;  - Save a mask, overlay, and measurement report.



6\. \*\*Validation and Analysis\*\*

&#x20;  - Inspect segmentation outputs.

&#x20;  - Compare estimated dimensions against physical reference measurements.

&#x20;  - Report measurement errors and limitations.

&#x20;  - Compare raw and undistorted image measurements.



7\. \*\*Documentation and Delivery\*\*

&#x20;  - Document installation, training, evaluation, and inference.

&#x20;  - Provide reproducible commands and project limitations.

&#x20;  - Commit project files and prepare the final demonstration.



\## Current Constraints

\- Metric scale calibration is provisional and requires further validation.

\- The current segmentation evaluation uses images of the same object and capture setup.

\- Measurements may be affected by camera pose, perspective, calibration quality, and reference alignment.



\## Deliverables

\- Camera calibration data and scripts.

\- Annotated dataset and split manifest.

\- Trained segmentation checkpoint.

\- Evaluation metrics and visualizations.

\- Image-to-measurement inference script.

\- Documentation and reproducible setup instructions.

