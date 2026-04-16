import mediapipe as mp
from new_features import normalize_and_extract_features
import csv 

def creating_PoseLandmark_instance():
  model_path = "datapipeline/../Model/pose_landmarker_full.task"

  '''
      #Giving shorter names to the classes and enums that we will be using from the mediapipe library.
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
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.IMAGE,
      num_poses=1
  )

  return PoseLandmarker.create_from_options(options)

def save_features_to_csv(all_features,csv_path):
  num_features = len(all_features[0][1])
  header = ["label"] + [f"feature_{i}" for i in range(num_features)]

  with open(csv_path,'w',newline='') as file:
    writer=csv.writer(file)
    writer.writerow(header)

    for label, feature_vector in all_features:
      row = [label] + feature_vector.tolist()
      writer.writerow(row)

if __name__ == "__main__":
  with creating_PoseLandmark_instance() as landmarker:
    feature_list = normalize_and_extract_features("Images",landmarker)
  
  save_features_to_csv(feature_list,"datapipeline/../dataset/pose_dataset.csv")