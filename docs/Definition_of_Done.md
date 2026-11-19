\# Definition of Done — MEASURE IT



The project is considered complete when the following requirements have been addressed and documented.



\## 1. Environment and Reproducibility

\- \[x] Python virtual environment created.

\- \[x] Required dependencies installed and pinned.

\- \[x] GPU availability verified.

\- \[ ] Installation and execution instructions documented.



\## 2. Camera Calibration

\- \[x] At least 20 checkerboard images processed.

\- \[x] Camera matrix and distortion coefficients saved.

\- \[x] Reprojection error calculated and recorded.

\- \[x] Undistortion script tested.

\- \[ ] Calibration accuracy limitations documented.



\## 3. Dataset

\- \[x] At least 70 images collected.

\- \[x] Segmentation masks prepared.

\- \[x] Training, validation, and test splits created.

\- \[x] Split manifest saved.

\- \[ ] Dataset and split integrity documented.



\## 4. Segmentation Model

\- \[x] DeepLabV3-ResNet50 trained.

\- \[x] Best model checkpoint saved.

\- \[x] Training history saved.

\- \[x] Test-set IoU, precision, and recall calculated.

\- \[x] Prediction visualizations generated and inspected.

\- \[ ] Additional requested average-precision metrics documented.



\## 5. Measurement Pipeline

\- \[x] Command-line image inference implemented.

\- \[x] Image undistortion integrated.

\- \[x] Segmentation integrated.

\- \[x] Pixel dimensions calculated.

\- \[x] Millimeter conversion implemented.

\- \[x] Mask, overlay, and JSON report generated.

\- \[ ] Reference-scale accuracy validated.

\- \[ ] Missing-object and reference-detection failure handling verified.



\## 6. Measurement Validation

\- \[ ] At least 10 independent physical measurements compared.

\- \[ ] Mean absolute error calculated.

\- \[ ] Mean percentage error calculated.

\- \[ ] Median and worst-case error reported.

\- \[ ] Raw versus undistorted measurements compared.

\- \[ ] Failure conditions and refusal rules documented.



\## 7. Delivery

\- \[ ] README completed.

\- \[ ] All required project files committed to Git.

\- \[ ] Reproduction instructions tested.

\- \[ ] Final demonstration recorded (maximum 5 minutes).

\- \[ ] Submission and pull request completed.



\## Known Limitations

The current pixel-to-millimeter scale is provisional. The current metric outputs must not be treated as validated physical measurements until independently checked against physical reference measurements. Segmentation performance does not establish measurement accuracy.

