\# MEASURE IT — Segmentation and Metric Measurement



A computer-vision prototype that segments a glasses case from an image, estimates its pixel dimensions, and converts those dimensions into provisional millimeter measurements.



\## Features



\- Checkerboard camera calibration and image undistortion.

\- Custom binary segmentation using DeepLabV3-ResNet50.

\- Training, validation, and test dataset splits.

\- Segmentation evaluation and prediction visualizations.

\- Command-line image inference.

\- Pixel-dimension estimation and provisional metric conversion.

\- Mask, overlay, and JSON report generation.



\## Project Structure



```text

measure-it/

├── calibration/

│   ├── images/

│   ├── scale\_images/

│   ├── calibrate.py

│   ├── camera\_calibration.npz

│   └── calculate\_scale.py

├── dataset/

│   ├── images/

│   │   ├── train/

│   │   ├── val/

│   │   └── test/

│   ├── masks/

│   ├── splits/

│   ├── prepare\_splits.py

│   └── annotate.py

├── docs/

├── inference/

│   ├── measure.py

│   └── results/

├── models/

│   ├── train\_segmentation.py

│   ├── evaluate.py

│   ├── visualize\_predictions.py

│   ├── best\_segmentation\_model.pth

│   └── training\_history.csv

├── requirements.txt

├── Plan.md

├── Objectives.md

├── Definition\_of\_Done.md

└── README.md

```



\## Requirements



\- Windows 10/11

\- Python 3.12

\- NVIDIA GPU with compatible CUDA-enabled PyTorch, or CPU

\- Git



\## Installation



Open a terminal in the project directory.



Create and activate a virtual environment:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



Install the pinned dependencies:



```cmd

python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

```



Verify the installation:



```cmd

python -m pip check

```



Check GPU availability:



```cmd

python -c "import torch; print('CUDA available:', torch.cuda.is\_available()); print(torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU mode')"

```



\## Camera Calibration



Checkerboard calibration images are stored in `calibration/images/`.



Run:



```cmd

python calibration/calibrate.py

```



The calibration process saves camera parameters and reprojection-error information in:



```text

calibration/camera\_calibration.npz

```



The calibration is specific to the camera configuration and image geometry used during capture.



\## Dataset



Images and corresponding masks are stored separately under `dataset/`.



The dataset split manifest is stored at:



```text

dataset/splits/split\_manifest.txt

```



The current dataset contains 84 images, divided into training, validation, and test subsets.



\## Model Training



Train the segmentation model with:



```cmd

python models/train\_segmentation.py

```



The best checkpoint and training history are saved in `models/`.



\## Model Evaluation



Run:



```cmd

python models/evaluate.py

```



Evaluation results are calculated on the held-out test set.



\## Prediction Visualization



Run:



```cmd

python models/visualize\_predictions.py

```



Generated visualizations are saved in:



```text

models/prediction\_visualizations/

```



\## Image Measurement



Run the inference pipeline on an image:



```cmd

python inference/measure.py dataset/images/test/20261119\_150227.jpg

```



The pipeline:



1\. Loads the input image.

2\. Applies camera undistortion.

3\. Predicts the glasses-case segmentation mask.

4\. Extracts the largest foreground component.

5\. Estimates pixel length and width.

6\. Converts pixel measurements using the configured provisional scale.

7\. Saves a segmentation mask, overlay, and JSON report.



Results are saved in:



```text

inference/results/

```



\## Example Output



```text

========== MEASUREMENT REPORT ==========

Image: example.jpg

Length: \[pixel measurement] pixels

Width:  \[pixel measurement] pixels

Length: \[provisional measurement] mm

Width:  \[provisional measurement] mm

Foreground probability: \[confidence]

Status: PROVISIONAL

```



\## Current Evaluation



Held-out test-set segmentation results:



| Metric | Result |

|---|---:|

| IoU | 91.32% |

| Precision | 96.83% |

| Recall | 94.13% |



Best validation IoU: 92.17%.



These metrics describe segmentation performance, not physical measurement accuracy.



\## Measurement Limitations



The current pixel-to-millimeter scale is provisional and has not passed independent physical validation. Initial ruler-derived scale estimates were inconsistent, and some resulting dimensions differ substantially from the approximate physical dimensions of the glasses case.



Therefore:



\- Pixel dimensions and segmentation outputs are the current reliable outputs.

\- Millimeter measurements must be treated as provisional.

\- Results depend on camera position, perspective, calibration quality, reference-plane alignment, and image conditions.

\- The prototype has been evaluated on images of one object and one capture setup.

\- Performance on different objects and camera configurations is not established.



\## Reproducibility



Use the pinned package versions in `requirements.txt`. Keep the calibration data, dataset split manifest, model checkpoint, and input image configuration with the project.



\## Project Status



This is a working prototype. Segmentation training, test evaluation, and command-line inference have been completed. Independent metric validation, additional requested evaluation metrics, and final delivery checks remain outstanding.



## Model Weights and Verification

**Model:** `models/best_segmentation_model.pth`

**Storage:** Git Large File Storage (Git LFS)

**SHA-256:**
``text
bbd7edb6b8c590159d05e530eaeacf28e80b03669668e5c15c0251cfe7ca4232
`` 
