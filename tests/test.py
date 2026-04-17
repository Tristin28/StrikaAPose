import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.search_engines.NearestNeighbours import SklearnSearchEngine
from src.dataset.load_csv import PoseClass
from src.predictor.predictor import predict_pose
from src.livepipeline.normalising_coords import normalize_live_coords
from src.datapipeline.config import DATASET_PATH

def main():
    search_engine = SklearnSearchEngine(model=NearestNeighbors(metric="euclidean"),k=3)

    pose_db = PoseClass(search_engine)
    pose_db.load_csv(DATASET_PATH)

    # fake example pose with 33 landmarks, each [x, y, z]
    raw_landmarks = np.random.rand(33, 3)

    feature_vector = normalize_live_coords(raw_landmarks)
    if feature_vector is None:
        print("Normalization failed")
        return

    label = predict_pose(feature_vector, pose_db)
    print("Predicted label:", label)

if __name__ == "__main__":
    main()