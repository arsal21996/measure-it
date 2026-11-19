\# Project Objectives — MEASURE IT



\## Primary Objective

Develop a computer-vision application that identifies a glasses case in an image and estimates its physical dimensions in millimeters.



\## Technical Objectives



1\. \*\*Camera Calibration\*\*

&#x20;  - Calibrate the camera using at least 20 checkerboard images.

&#x20;  - Store camera intrinsic parameters and lens distortion coefficients.

&#x20;  - Measure and document calibration reprojection error.



2\. \*\*Image Processing\*\*

&#x20;  - Undistort input images before measurement.

&#x20;  - Preserve the original image and generate corrected outputs.



3\. \*\*Dataset\*\*

&#x20;  - Prepare at least 70 captured images.

&#x20;  - Create corresponding binary segmentation masks.

&#x20;  - Maintain separate training, validation, and test sets.

&#x20;  - Keep a reproducible split manifest.



4\. \*\*Deep Learning\*\*

&#x20;  - Train a custom DeepLabV3-ResNet50 segmentation model.

&#x20;  - Save the best model checkpoint.

&#x20;  - Evaluate using Intersection over Union (IoU), precision, and recall.

&#x20;  - Report the requested average-precision metrics where implemented.



5\. \*\*Automated Measurement\*\*

&#x20;  - Accept a single image through a command-line interface.

&#x20;  - Segment the glasses case.

&#x20;  - Estimate its pixel length and width.

&#x20;  - Convert pixel measurements to millimeters using a reference scale.

&#x20;  - Generate a segmentation mask, visual overlay, and structured report.

&#x20;  - Handle missing or undetected objects gracefully.



6\. \*\*Validation\*\*

&#x20;  - Compare predicted dimensions with physical measurements.

&#x20;  - Calculate mean absolute error and percentage error.

&#x20;  - Identify sources of measurement uncertainty.

&#x20;  - Document conditions under which measurements should not be trusted.



\## Success Criteria

\- The application runs using the documented environment.

\- The trained model produces usable segmentation masks on held-out test images.

\- The inference command processes an individual image successfully.

\- Results include dimensions, confidence information, and visual outputs.

\- Calibration and measurement limitations are clearly disclosed.

\- Another user can reproduce the workflow from the documentation.



\## Scope

The current prototype focuses on one glasses-case object and the capture setup used to create the dataset. Generalization to other objects, cameras, distances, and lighting conditions is outside the validated scope.

