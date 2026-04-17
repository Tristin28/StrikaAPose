import numpy as np
from src.datapipeline.preprocessing_coords import position, scaling, rotate_pose
from src.datapipeline.new_features import build_feature_vector

def normalize_live_coords(raw_landmarks):
    if raw_landmarks is None:
        print("NO LANDMARKS WERE RECEIVED")
        return None
    
    coords = np.array(raw_landmarks, dtype=np.float64)
    if coords.shape != (33, 3):
        print("INVALID LANDMARK SHAPE")
        return None

    coords = position(coords)
    
    coords = scaling(coords)
    if coords is None:
        return None
    
    coords = rotate_pose(coords)

    feature_vector = build_feature_vector(coords)
    return feature_vector