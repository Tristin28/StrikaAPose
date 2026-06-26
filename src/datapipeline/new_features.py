import numpy as np
from src.datapipeline.preprocessing_coords import extracting_raw_coords, normalise_single_pose
from src.datapipeline.config import PROJECT_ROOT

LANDMARKS = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "NOSE": 0,
    #coarse hand points that MediaPipe Pose already gives us (no extra model needed)
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22
}

POSE_RELATIONSHIP_WEIGHT = 2.0


def normalize_and_extract_features(images_folder, landmarker, stats=None):
    extracted = extracting_raw_coords(images_folder, landmarker, stats)

    all_features = []
    for file_path, pose_landmarks in extracted:
        coords = normalise_single_pose(pose_landmarks)
        if coords is None:
            if stats is not None:
                stats["normalisation_failed"] += 1
            print(f"[COULD NOT NORMALISE]: {file_path}")
            continue

        feature_vector = build_feature_vector(coords)
        label = file_path.parent.name
        image_path = file_path.relative_to(PROJECT_ROOT).as_posix() #e.g. "Images/Heart/Heart_001.jpg" so the frontend can load it
        if stats is not None:
            stats["added_to_dataset"] += 1
        all_features.append((image_path, label, feature_vector))

    return all_features

def extract_joint_angles(coords):
    angles = [
        compute_angle(coords[LANDMARKS["LEFT_SHOULDER"]], coords[LANDMARKS["LEFT_ELBOW"]], coords[LANDMARKS["LEFT_WRIST"]]),
        compute_angle(coords[LANDMARKS["RIGHT_SHOULDER"]], coords[LANDMARKS["RIGHT_ELBOW"]], coords[LANDMARKS["RIGHT_WRIST"]]),
        compute_angle(coords[LANDMARKS["LEFT_ELBOW"]], coords[LANDMARKS["LEFT_SHOULDER"]], coords[LANDMARKS["LEFT_HIP"]]),
        compute_angle(coords[LANDMARKS["RIGHT_ELBOW"]], coords[LANDMARKS["RIGHT_SHOULDER"]], coords[LANDMARKS["RIGHT_HIP"]]),
        compute_angle(coords[LANDMARKS["LEFT_SHOULDER"]], coords[LANDMARKS["LEFT_HIP"]], coords[LANDMARKS["LEFT_KNEE"]]),
        compute_angle(coords[LANDMARKS["RIGHT_SHOULDER"]], coords[LANDMARKS["RIGHT_HIP"]], coords[LANDMARKS["RIGHT_KNEE"]]),
        compute_angle(coords[LANDMARKS["LEFT_HIP"]], coords[LANDMARKS["LEFT_KNEE"]], coords[LANDMARKS["LEFT_ANKLE"]]),
        compute_angle(coords[LANDMARKS["RIGHT_HIP"]], coords[LANDMARKS["RIGHT_KNEE"]], coords[LANDMARKS["RIGHT_ANKLE"]]),
    ]

    return np.array(angles, dtype=np.float64)

def extract_key_distances(coords):
    shoulder_center = (coords[LANDMARKS["LEFT_SHOULDER"]] + coords[LANDMARKS["RIGHT_SHOULDER"]]) / 2

    distances = [
        np.linalg.norm(coords[LANDMARKS["LEFT_WRIST"]] - coords[LANDMARKS["RIGHT_WRIST"]]),
        np.linalg.norm(coords[LANDMARKS["LEFT_WRIST"]] - coords[LANDMARKS["NOSE"]]),
        np.linalg.norm(coords[LANDMARKS["RIGHT_WRIST"]] - coords[LANDMARKS["NOSE"]]),
        np.linalg.norm(coords[LANDMARKS["LEFT_WRIST"]] - shoulder_center),
        np.linalg.norm(coords[LANDMARKS["RIGHT_WRIST"]] - shoulder_center),
        np.linalg.norm(coords[LANDMARKS["LEFT_ANKLE"]] - shoulder_center),
        np.linalg.norm(coords[LANDMARKS["RIGHT_ANKLE"]] - shoulder_center),
    ]

    return np.array(distances, dtype=np.float64)

def extract_pose_relationship_features(coords):
    '''
        Pose-specific relationships used to separate visually similar arm poses.
        These are still generic numeric features, not hard-coded class rules.
    '''
    left_shoulder = coords[LANDMARKS["LEFT_SHOULDER"]]
    right_shoulder = coords[LANDMARKS["RIGHT_SHOULDER"]]
    left_elbow = coords[LANDMARKS["LEFT_ELBOW"]]
    right_elbow = coords[LANDMARKS["RIGHT_ELBOW"]]
    left_wrist = coords[LANDMARKS["LEFT_WRIST"]]
    right_wrist = coords[LANDMARKS["RIGHT_WRIST"]]
    left_hip = coords[LANDMARKS["LEFT_HIP"]]
    right_hip = coords[LANDMARKS["RIGHT_HIP"]]
    nose = coords[LANDMARKS["NOSE"]]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    elbow_center = (left_elbow + right_elbow) / 2
    wrist_center = (left_wrist + right_wrist) / 2

    left_wrist_to_left_hip = distance(left_wrist, left_hip)
    right_wrist_to_right_hip = distance(right_wrist, right_hip)
    left_wrist_to_left_shoulder = distance(left_wrist, left_shoulder)
    right_wrist_to_right_shoulder = distance(right_wrist, right_shoulder)

    features = [
        # Hands-on-hips evidence: wrists should be close to the same-side hips.
        left_wrist_to_left_hip,
        right_wrist_to_right_hip,
        distance(left_wrist, right_hip),
        distance(right_wrist, left_hip),
        distance(left_elbow, left_hip),
        distance(right_elbow, right_hip),

        # Raised-arm evidence: wrists and elbows move above the shoulders.
        left_wrist[1] - left_shoulder[1],
        right_wrist[1] - right_shoulder[1],
        left_elbow[1] - left_shoulder[1],
        right_elbow[1] - right_shoulder[1],
        wrist_center[1] - shoulder_center[1],
        elbow_center[1] - shoulder_center[1],
        max(0.0, left_shoulder[1] - left_wrist[1]),
        max(0.0, right_shoulder[1] - right_wrist[1]),
        max(0.0, left_shoulder[1] - left_elbow[1]),
        max(0.0, right_shoulder[1] - right_elbow[1]),

        # Heart-pose evidence: wrists tend to move close together near the face.
        distance(left_wrist, right_wrist),
        distance(wrist_center, nose),
        distance(wrist_center, shoulder_center),
        abs(left_wrist[1] - right_wrist[1]),
        abs(left_elbow[1] - right_elbow[1]),

        # Body-relative hand position: useful when poses differ mostly by hand placement.
        left_wrist[1] - left_hip[1],
        right_wrist[1] - right_hip[1],
        wrist_center[1] - hip_center[1],
        left_wrist[0] - left_hip[0],
        right_wrist[0] - right_hip[0],
        left_wrist[0] - left_shoulder[0],
        right_wrist[0] - right_shoulder[0],
        abs(left_wrist[0] - right_wrist[0]),
        abs(left_elbow[0] - right_elbow[0]),

        # Ratios reduce sensitivity to person size after shoulder-width scaling.
        safe_ratio(left_wrist_to_left_hip, left_wrist_to_left_shoulder),
        safe_ratio(right_wrist_to_right_hip, right_wrist_to_right_shoulder),
    ]

    return np.array(features, dtype=np.float64) * POSE_RELATIONSHIP_WEIGHT

def extract_hand_features(coords):
    '''
        Lightweight hand-orientation features from the coarse hand points MediaPipe Pose
        already provides (wrist, thumb, index, pinky). No extra hand model is used.
        Per hand we capture: how open the hand is, how the wrist is bent, and where the
        thumb points relative to the index finger.
    '''
    hand_features = [
        #left hand
        compute_angle(coords[LANDMARKS["LEFT_INDEX"]], coords[LANDMARKS["LEFT_WRIST"]], coords[LANDMARKS["LEFT_PINKY"]]),   #hand spread (index-wrist-pinky)
        compute_angle(coords[LANDMARKS["LEFT_ELBOW"]], coords[LANDMARKS["LEFT_WRIST"]], coords[LANDMARKS["LEFT_INDEX"]]),    #wrist flexion vs forearm
        compute_angle(coords[LANDMARKS["LEFT_THUMB"]], coords[LANDMARKS["LEFT_WRIST"]], coords[LANDMARKS["LEFT_INDEX"]]),    #thumb vs index
        #right hand
        compute_angle(coords[LANDMARKS["RIGHT_INDEX"]], coords[LANDMARKS["RIGHT_WRIST"]], coords[LANDMARKS["RIGHT_PINKY"]]),
        compute_angle(coords[LANDMARKS["RIGHT_ELBOW"]], coords[LANDMARKS["RIGHT_WRIST"]], coords[LANDMARKS["RIGHT_INDEX"]]),
        compute_angle(coords[LANDMARKS["RIGHT_THUMB"]], coords[LANDMARKS["RIGHT_WRIST"]], coords[LANDMARKS["RIGHT_INDEX"]]),
    ]

    return np.array(hand_features, dtype=np.float64)

def distance(a, b):
    return np.linalg.norm(a - b)

def safe_ratio(numerator, denominator):
    if denominator < 1e-8:
        return 0.0
    return numerator / denominator

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    euclidian_distance = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if euclidian_distance < 1e-8:
        '''
            if cooedinates (b, a) or (b, c) are near each other or the same point then this means
            There is no reliable line to use to compute angle b
        '''
        return 0.0
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    

def build_feature_vector(coords):
    flat_coords = coords.flatten() #66 features
    angles = extract_joint_angles(coords) #8 features
    distances = extract_key_distances(coords) #7 features
    hand_features = extract_hand_features(coords) #6 features (coarse hand orientation)
    pose_relationships = extract_pose_relationship_features(coords) #32 features (pose-specific arm/hand placement)

    return np.concatenate([flat_coords, angles, distances, hand_features, pose_relationships])
