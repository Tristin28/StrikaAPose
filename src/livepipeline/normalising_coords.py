import numpy as np
from src.datapipeline.preprocessing_coords import position, scaling, rotate_pose
from src.datapipeline.new_features import build_feature_vector

def normalize_live_coords(raw_landmarks):
    coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in raw_landmarks])

    coords = position(coords)
    
    coords = scaling(coords)
    if coords is None:
        return None
    
    coords = rotate_pose(coords)

    feature_vector = build_feature_vector(coords)
    return feature_vector