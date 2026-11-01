# Shape Classification

A Python computer-vision experiment comparing Decision Tree, Random Forest, and RBF SVM classifiers on geometric shapes. OpenCV extracts contour features: area, perimeter, aspect ratio, circularity, vertex count, and seven Hu moments.

## Setup

Use Python 3 and run commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install:

```sh
python -m pip install numpy opencv-python scikit-learn
```

No pinned dependency file is included.

## Train and evaluate

```sh
python evaluation.py
```

This loads the checked-in training and test images, extracts and scales features, trains all three classifiers, and prints accuracy and per-class reports. Importing `training.py` triggers training, so a separate training command is unnecessary for evaluation.

## Project layout

| Path | Purpose |
| --- | --- |
| `features_updated.py` | Inverted thresholding, contour features, dataset loading, and feature scaling |
| `training.py` | Decision Tree, 100-tree Random Forest, and RBF SVM fitting |
| `evaluation.py` | Predictions, accuracy, classification reports, and model comparison |
| `features.py` | Earlier feature-extraction version; unused by the current training entry point |
| `processed_dataset_split/train/`, `processed_dataset_split/test/` | Images organized into 17 shape-class folders |
| `selected_dataset/` | Additional selected images |

The scaler fits the training features and transforms the test features. Class labels come from folder names. Keep the same feature extraction and class folders when changing the data.

## Limits and changes

This is an experiment script with no GUI, saved model export, or standalone prediction API. Images that cannot produce usable contours may stop processing or be skipped. Results depend on the checked-in split; the script does not establish how independent the source images are.

When contributing, describe changes to image preprocessing, class folders, or splits, and rerun `python evaluation.py` to compare all three models.
