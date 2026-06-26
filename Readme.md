# Strike A Pose

Strike A Pose is a Flask web app that compares a live webcam pose against a small dataset of labelled pose images. The browser detects body landmarks with MediaPipe, sends those landmarks to the Flask backend, and the backend returns the closest confident pose match from the dataset.

Python 3.12 or newer is recommended for this project.

## Project Structure

```text
.
├── Images/                         # Labelled dataset images, grouped by pose/class name
│   ├── Heart/
│   ├── HandsOnHips/
│   ├── ArmsRaised/
│   ├── danielsturridge/
│   └── luissuarez/
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
│   │   └── normalising_coords.py    # Normalises live landmarks and checks live pose activity
│   ├── predictor/
│   │   └── predictor.py             # Applies nearest-neighbour voting and confidence guards
│   └── search_engines/
│       ├── interface.py             # Search engine interface
│       └── NearestNeighbours.py     # scikit-learn nearest-neighbour search implementation
├── static/
│   ├── app.js                       # Browser webcam, MediaPipe, and result UI logic
│   └── style.css                    # Frontend styling
├── templates/
│   └── index.html                   # Main Flask-rendered page
└── requirements.txt                 # Python dependencies
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

Allow camera access when the browser asks. The page will show the live webcam feed, draw the detected skeleton, and send the current pose to the backend every second.

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

The evaluator loads `src/dataset/pose_dataset.csv` and uses stratified cross-validation. The dataset is split into multiple folds while keeping the pose labels as balanced as possible in each split. For each metric, the script repeatedly fits the nearest-neighbour model on part of the dataset and tests it on the remaining part.

The output includes:

```text
Top-1 Accuracy
Top-k Accuracy
Average correct-match distance
Average wrong-match distance
Most common label mistakes
```

Top-1 accuracy means the final predicted label was correct. Top-k accuracy means the correct label appeared anywhere in the nearest neighbours being checked. Labels with fewer than two examples are excluded because cross-validation cannot fairly test a class that has no remaining training example when its only image is placed in the test fold.

Optional parameters can be changed from the terminal:

```bash
python -m src.evaluation.evaluate_metrics --folds 5 --k 3
```

This is useful for testing different values of `k` or different cross-validation settings without editing the source code.

The evaluation results describe performance on the generated CSV dataset. Live webcam matching can behave differently because the live camera introduces different cropping, camera angle, mirroring, lighting, and landmark noise. For that reason, the live app retrieves the top 5 neighbours and uses majority voting for a more stable result.

## How The App Works

1. `static/app.js` opens the webcam and loads the MediaPipe pose model from `/Model/pose_landmarker_full.task`.
2. MediaPipe detects 33 body landmarks in the browser. The frontend sends the x/y landmark coordinates, landmark visibility values, and selected metric to Flask.
3. Every second, the latest visible pose is sent to Flask through the `/predict` endpoint.
4. `src/app.py` rejects frames where key landmarks such as shoulders, elbows, wrists, or hips are not visible enough.
5. `src/livepipeline/normalising_coords.py` normalises the live landmarks using the same feature logic as the dataset builder and checks whether the pose is active enough to avoid matching a neutral standing pose.
6. `src/predictor/predictor.py` searches `src/dataset/pose_dataset.csv` with the selected metric: Euclidean, cosine, or Manhattan.
7. The backend retrieves the top 5 nearest neighbours. If at least 3 of those neighbours agree, that label wins by majority vote. A strong 4-out-of-5 vote can pass a slightly relaxed live-camera distance threshold.
8. Flask returns the predicted label, closest image from the winning label, distance, and debug neighbours to the browser.

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
