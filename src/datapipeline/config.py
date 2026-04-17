'''
    This is just to store the file paths of the model and csv
'''
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "Model" / "pose_landmarker_full.task"
DATASET_PATH = PROJECT_ROOT / "dataset" / "pose_dataset.csv"