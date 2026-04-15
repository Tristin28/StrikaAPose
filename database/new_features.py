import numpy as np
from preprocessing_coords import extracting_raw_coords, normalise_single_pose

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
NOSE = 0

def normalize_and_extract_features(images_folder, landmarker):
    extracted = extracting_raw_coords(images_folder, landmarker)

    all_features = []
    for file_path, pose_landmarks in extracted:
        coords = normalise_single_pose(pose_landmarks)
        if coords is None:
            print(f"[COULD NOT NORMALISE]: {file_path}")
            continue

        feature_vector = build_feature_vector(coords)
        
        all_features.append((file_path, feature_vector))

    return all_features

def extract_joint_angles(coords):
    angles = [
        compute_angle(coords[LEFT_SHOULDER], coords[LEFT_ELBOW], coords[LEFT_WRIST]),
        compute_angle(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST]),
        compute_angle(coords[LEFT_ELBOW], coords[LEFT_SHOULDER], coords[LEFT_HIP]),
        compute_angle(coords[RIGHT_ELBOW], coords[RIGHT_SHOULDER], coords[RIGHT_HIP]),
        compute_angle(coords[LEFT_SHOULDER], coords[LEFT_HIP], coords[LEFT_KNEE]),
        compute_angle(coords[RIGHT_SHOULDER], coords[RIGHT_HIP], coords[RIGHT_KNEE]),
        compute_angle(coords[LEFT_HIP], coords[LEFT_KNEE], coords[LEFT_ANKLE]),
        compute_angle(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE]),
    ]

    return np.array(angles, dtype=np.float64)

def extract_key_distances(coords):
    shoulder_center = (coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2

    distances = [
        np.linalg.norm(coords[LEFT_WRIST] - coords[RIGHT_WRIST]),
        np.linalg.norm(coords[LEFT_WRIST] - coords[NOSE]),
        np.linalg.norm(coords[RIGHT_WRIST] - coords[NOSE]),
        np.linalg.norm(coords[LEFT_WRIST] - shoulder_center),
        np.linalg.norm(coords[RIGHT_WRIST] - shoulder_center),
        np.linalg.norm(coords[LEFT_ANKLE] - shoulder_center),
        np.linalg.norm(coords[RIGHT_ANKLE] - shoulder_center),
    ]

    return np.array(distances, dtype=np.float64)

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    

def build_feature_vector(coords):
    flat_coords = coords.flatten() #99 features
    angles = extract_joint_angles(coords) #8 features
    distances = extract_key_distances(coords) #7 features

    return np.concatenate([flat_coords, angles, distances])
