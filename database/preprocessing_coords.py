import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
from database.processing_images import resize_if_small

def create_mediapipe_object(image, landmarker):
    '''
        Passed image has to be in RGB format for mediapipe, and this function will return the 33 landmarks or nothing
    '''
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
    except Exception as e:
        print(f"{e}")
        return None

    try:
        result = landmarker.detect(mp_image)
    except Exception as e:
        print(f"\n[DETECTION PROCESS FAILED]: {e}")
        return None

    if not result.pose_landmarks:
        '''
            Checking it still because the model can still run successfully but no person was detected
        '''
        print(f"\nNO PERSON DETECTED")
        return None
    
    pose_landmarks = result.pose_landmarks[0] #Storing 33 landmark objects where each object contains 4 fields which are equivelant to the 3D coordinates + visibility

    visibility_count = sum(1 for lm in pose_landmarks if hasattr(lm, "visibility") and lm.visibility >= 0.5)
    if visibility_count < 15:
        print(f"\nTOO FEW VISIBLE LANDMARKS: {visibility_count}")
        return None

    return pose_landmarks

def extracting_raw_coords(images_folder, landmarker):
    images_dir = Path(images_folder).resolve()
    extracted_poses_lm = []

    for file_path in images_dir.rglob("*"):
        if not file_path.is_file():
            continue

        img = cv2.imread(str(file_path))

        if img is None:
            print(f"LOAD FAILED: {file_path}")
            continue

        img = resize_if_small(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose_landmarks = create_mediapipe_object(img_rgb, landmarker)

        if pose_landmarks is None:
            print(f"SKIPPED: {file_path}\n")
            continue

        extracted_poses_lm.append((file_path, pose_landmarks))
        print(f"POSE EXTRACTED: {file_path}")

    return extracted_poses_lm


def position(coords,pose_landmarks,LEFT_SHOULDER=11, RIGHT_SHOULDER=12):
    '''
       Landmarks' values would now be representing coordinates w.r.t a new origin which is the centre of shoulder 
       (chose shoulder centre to be the origin because it is the body part which is almost always visible)
    '''

    if pose_landmarks[LEFT_SHOULDER].visibility > 0.5 and pose_landmarks[RIGHT_SHOULDER].visibility >0.5:
        return coords - ((coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2)
    else:
        print("SKIPPED\n")
        return None
    
def scaling(coords, LEFT_SHOULDER=11, RIGHT_SHOULDER=12):
    #norm function calculates Euclidian distance on the passed coordinates (vectors)
    shoulder_width = np.linalg.norm(coords[LEFT_SHOULDER] - coords[RIGHT_SHOULDER])
    
    if shoulder_width > 0:
        return coords / shoulder_width
    else:
        print("INVALID SHOULDER WIDTH")
        return None
    

def rotate_pose(coords, LEFT_SHOULDER=11, RIGHT_SHOULDER=12):
    roated_coords = coords.copy()

    shoulder_vector = coords[LEFT_SHOULDER] - coords[RIGHT_SHOULDER]

    angle = np.arctan2(shoulder_vector[0],shoulder_vector[1]) #Returning the angle of which direction the shoulder is pointing to i.e. returning 
    
    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)

    x = coords[:, 0]
    y = coords[:, 1]

    #rotating the x-y axis by the shoulder angle and leaving z axist alone because i fixed the sideways tilt in the frames.
    roated_coords[:, 0] = x * cos_theta - y * sin_theta
    roated_coords[:, 1] = x * sin_theta + y * cos_theta
    roated_coords[:, 2] = coords[:, 2]

    return roated_coords

def normalise_single_pose(pose_landmarks):
    '''
        will normalsie the pose so that the coordinates are relative to the actual pose and not dependent on the image features 
    '''

    coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])

    coords = position(coords,pose_landmarks)
    if coords is None:
        return None
    
    coords = scaling(coords)
    if coords is None:
        return None
    
    coords = rotate_pose(coords)
    return coords