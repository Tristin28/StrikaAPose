import numpy as np
from src.datapipeline.preprocessing_coords import position, scaling, rotate_pose
from src.datapipeline.new_features import LANDMARKS, build_feature_vector, compute_angle, distance

def normalize_live_pose(raw_landmarks):
    if raw_landmarks is None:
        print("NO LANDMARKS WERE RECEIVED")
        return None
    
    coords = np.array(raw_landmarks, dtype=np.float64)
    if coords.shape != (33, 2):
        print("INVALID LANDMARK SHAPE")
        return None

    coords = position(coords)
    
    coords = scaling(coords)
    if coords is None:
        return None
    
    coords = rotate_pose(coords)

    feature_vector = build_feature_vector(coords)
    return coords, feature_vector

def normalize_live_coords(raw_landmarks):
    normalized_pose = normalize_live_pose(raw_landmarks)
    if normalized_pose is None:
        return None

    _, feature_vector = normalized_pose
    return feature_vector

def assess_live_pose_activity(coords):
    left_shoulder = coords[LANDMARKS["LEFT_SHOULDER"]]
    right_shoulder = coords[LANDMARKS["RIGHT_SHOULDER"]]
    left_elbow = coords[LANDMARKS["LEFT_ELBOW"]]
    right_elbow = coords[LANDMARKS["RIGHT_ELBOW"]]
    left_wrist = coords[LANDMARKS["LEFT_WRIST"]]
    right_wrist = coords[LANDMARKS["RIGHT_WRIST"]]
    left_hip = coords[LANDMARKS["LEFT_HIP"]]
    right_hip = coords[LANDMARKS["RIGHT_HIP"]]
    nose = coords[LANDMARKS["NOSE"]]

    wrist_center = (left_wrist + right_wrist) / 2
    left_wrist_above_shoulder = float(left_shoulder[1] - left_wrist[1])
    right_wrist_above_shoulder = float(right_shoulder[1] - right_wrist[1])
    left_elbow_above_shoulder = float(left_shoulder[1] - left_elbow[1])
    right_elbow_above_shoulder = float(right_shoulder[1] - right_elbow[1])
    left_elbow_angle = float(compute_angle(left_shoulder, left_elbow, left_wrist))
    right_elbow_angle = float(compute_angle(right_shoulder, right_elbow, right_wrist))

    wrists_together = float(distance(left_wrist, right_wrist))
    wrist_center_to_nose = float(distance(wrist_center, nose))
    left_wrist_to_hip = float(distance(left_wrist, left_hip))
    right_wrist_to_hip = float(distance(right_wrist, right_hip))
    left_elbow_out = float(abs(left_elbow[0] - left_shoulder[0]))
    right_elbow_out = float(abs(right_elbow[0] - right_shoulder[0]))

    arms_raised = bool(
        left_wrist_above_shoulder > 0.45
        or right_wrist_above_shoulder > 0.45
        or left_elbow_above_shoulder > 0.25
        or right_elbow_above_shoulder > 0.25
    )
    hands_on_hips = bool(
        left_wrist_to_hip < 1.1
        and right_wrist_to_hip < 1.1
        and (left_elbow_angle < 2.6 or right_elbow_angle < 2.6)
    )
    heart_like = bool(wrists_together < 1.15 and wrist_center_to_nose < 1.75)
    bent_arm_pose = bool(
        (left_elbow_angle < 2.25 or right_elbow_angle < 2.25)
        and (left_elbow_out > 0.45 or right_elbow_out > 0.45)
    )

    return {
        "active": bool(arms_raised or hands_on_hips or heart_like or bent_arm_pose),
        "arms_raised": arms_raised,
        "hands_on_hips": hands_on_hips,
        "heart_like": heart_like,
        "bent_arm_pose": bent_arm_pose,
    }
