from collections import Counter
import mediapipe as mp
from src.datapipeline.new_features import normalize_and_extract_features
import csv 
from src.datapipeline.config import DATASET_PATH, MODEL_PATH

def creating_PoseLandmark_instance(model_path=MODEL_PATH):
  '''
      Giving shorter names to the classes and enums that we will be using from the mediapipe library.
  '''
  BaseOptions = mp.tasks.BaseOptions #A class where its instance will have its fields initialised to the location (file path) of the model.
  PoseLandmarker = mp.tasks.vision.PoseLandmarker
  PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode #Enum for the running mode of the landmarker. It can be either IMAGE, VIDEO or LIVE_STREAM.

  '''
      An instance which its fields are initialised with the respective configurations which come from other instances and enums.
      Note further configurations can be added either later on or directly in the constructor of the PoseLandmarkerOptions class.
  '''
  options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=str(model_path)),
      running_mode=VisionRunningMode.IMAGE,
      num_poses=1
  )

  return PoseLandmarker.create_from_options(options)

def save_features_to_csv(all_features,csv_path):
  if not all_features:
    raise ValueError("No valid pose features were extracted, so the dataset CSV was not written.")

  num_features = len(all_features[0][2])
  header = ["image_path", "label"] + [f"feature_{i}" for i in range(num_features)]

  with open(csv_path,'w',newline='') as file:
    writer=csv.writer(file)
    writer.writerow(header)

    for image_path, label, feature_vector in all_features:
      row = [image_path, label] + feature_vector.tolist()
      writer.writerow(row)

def print_build_summary(stats):
  skipped = stats["images_seen"] - stats["added_to_dataset"]

  print("\nDataset build summary")
  print("---------------------")
  print(f"Images scanned: {stats['images_seen']}")
  print(f"Images added to dataset: {stats['added_to_dataset']}")
  print(f"Images skipped: {skipped}")

  skip_reasons = [
    ("load_failed", "Could not load image"),
    ("invalid_mediapipe_image", "Could not convert image for MediaPipe"),
    ("detection_failed", "MediaPipe detection failed"),
    ("no_person_detected", "No person detected"),
    ("too_few_visible_landmarks", "Too few visible landmarks"),
    ("normalisation_failed", "Pose normalisation failed"),
  ]

  if skipped > 0:
    print("\nSkipped image reasons:")
    for key, label in skip_reasons:
      if stats[key]:
        print(f"- {label}: {stats[key]}")

if __name__ == "__main__":
  build_stats = Counter()
  with creating_PoseLandmark_instance() as landmarker:
    feature_list = normalize_and_extract_features("Images", landmarker, build_stats)
  
  save_features_to_csv(feature_list,DATASET_PATH)
  print_build_summary(build_stats)
