import numpy as np
from datapipeline.preprocessing_coords import position, scaling, rotate_pose
from datapipeline.new_features import build_feature_vector

def normalize_live_coords(raw_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])

    coords = position(coords,raw_landmarks)
    if coords is None:
        return None
    
    coords = scaling(coords)
    if coords is None:
        return None
    
    coords = rotate_pose(coords)

    feature_vector = build_feature_vector(coords)
    return feature_vector