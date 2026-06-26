# Strike A Pose

Strike A Pose is a Flask web app that compares a live webcam pose against a small dataset of labelled pose images. The browser detects body landmarks with MediaPipe, sends those landmarks to the Flask backend, and the backend returns the closest matching pose from the dataset.

Python 3.12 or newer is recommended for this project.

## Project Structure

```text
.
├── Images/                         # Labelled dataset images, grouped by pose/class name
│   ├── Heart/
│   ├── kneeslide/
│   ├── luissuarez/
│   └── pointing/
├── Model/
│   └── pose_landmarker_full.task    # MediaPipe pose model used by the app and dataset builder
├── src/
│   ├── app.py                       # Flask app, routes, model serving, and /predict endpoint
│   ├── datapipeline/
│   │   ├── Build_CSV.py             # Rebuilds the pose feature CSV from Images/
│   │   ├── config.py                # Shared project, model, and dataset paths
│   │   ├── new_features.py          # Builds feature vectors from normalized pose landmarks
│   │   ├── preprocessing_coords.py  # Pose landmark extraction, positioning, scaling, rotation
│   │   └── processing_images.py     # Image processing helpers
│   ├── dataset/
│   │   ├── load_csv.py              # Loads pose_dataset.csv and fits search engines
│   │   └── pose_dataset.csv         # Generated feature dataset used by the backend
│   ├── evaluation/
│   │   └── evaluate_metrics.py      # Cross-validation script for comparing distance metrics
│   ├── livepipeline/
│   │   └── normalising_coords.py    # Normalizes live browser landmarks before prediction
│   ├── predictor/
│   │   └── predictor.py             # Finds the closest match and predicted pose label
│   └── search_engines/
│       ├── interface.py             # Search engine interface
│       └── NearestNeighbours.py     # scikit-learn nearest-neighbour search implementation
├── static/
│   ├── app.js                       # Browser webcam, MediaPipe, and result UI logic
│   └── style.css                    # Frontend styling
├── templates/
│   └── index.html                   # Main Flask-rendered page
├── tests/                           # Test files
├── requirements.txt                 # Python dependencies
└── webprototype                     # Older standalone prototype, not required by the Flask app
```

## Setup

From the project root, create or activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The app also loads MediaPipe Tasks Vision in the browser from a CDN, so the browser needs internet access the first time it loads the page.

## Run The Web App

Start the Flask app from the project root:

```bash
python -m src.app
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000
```

Allow camera access when the browser asks. The page will show the live webcam feed, draw the detected skeleton, and update the closest dataset match every 5 seconds.

## Rebuild The Dataset

The generated dataset is stored at:

```text
src/dataset/pose_dataset.csv
```

Rebuild it whenever you add, remove, rename, or replace images in `Images/`.

```bash
python -m src.datapipeline.Build_CSV
```

To customise the dataset for your own poses:

1. Add a folder under `Images/` for each pose label.
2. Put the related images inside that folder.
3. Rebuild the CSV with `python -m src.datapipeline.Build_CSV`.
4. Run the web app again with `python -m src.app`.

Example:

```text
Images/
├── celebration/
│   ├── image_001.jpg
│   └── image_002.jpg
└── pointing/
    ├── image_001.jpg
    └── image_002.jpg
```

The folder name becomes the pose label returned by the predictor.

When the dataset is rebuilt, the script prints a build summary showing how many image files were scanned, how many were added to `pose_dataset.csv`, and how many were skipped. Skipped images are grouped by reason, such as no person detected, too few visible landmarks, or pose normalisation failure.

## Evaluate Similarity Metrics

The web app supports three K-nearest-neighbour distance metrics:

```text
euclidean
cosine
manhattan
```

To compare their performance on the generated dataset, run:

```bash
python -m src.evaluation.evaluate_metrics
```

The evaluator loads `src/dataset/pose_dataset.csv` and uses stratified cross-validation. The dataset is split into multiple folds while keeping the pose labels as balanced as possible in each split. For each metric, the script repeatedly trains the nearest-neighbour model on part of the dataset and tests it on the remaining part.

The output includes:

```text
Top-1 Accuracy
Top-3 Accuracy
Average correct-match distance
Average wrong-match distance
Most common label mistakes
```

Top-1 accuracy means the final majority-vote prediction was correct. Top-3 accuracy means the correct label appeared anywhere in the three nearest neighbours. Labels with fewer than two examples are excluded because cross-validation cannot fairly test a class that has no remaining training example when its only image is placed in the test fold.

Optional parameters can be changed from the terminal:

```bash
python -m src.evaluation.evaluate_metrics --folds 5 --k 3
```

This is useful for testing different values of `k` or different cross-validation settings without editing the source code.

## How The App Works

1. `static/app.js` opens the webcam and loads the MediaPipe pose model from `/Model/pose_landmarker_full.task`.
2. MediaPipe detects 33 body landmarks in the browser.
3. Every 5 seconds, the latest landmarks are sent to Flask through the `/predict` endpoint.
4. `src/livepipeline/normalising_coords.py` normalizes the live landmarks using the same feature logic as the dataset builder.
5. `src/predictor/predictor.py` searches `src/dataset/pose_dataset.csv` with the selected metric: Euclidean, cosine, or Manhattan.
6. Flask returns the closest matching image and predicted pose label to the browser.

## Useful Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild the generated pose dataset
python -m src.datapipeline.Build_CSV

# Evaluate Euclidean, cosine, and Manhattan matching
python -m src.evaluation.evaluate_metrics

# Run the Flask web app
python -m src.app
```
