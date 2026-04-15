import mediapipe as mp
from database.new_features import normalize_and_extract_features
import csv 

model_path = '../Model/pose_landmarker_full.task' #Currently in the same directory as this file.

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

with PoseLandmarker.create_from_options(options) as landmarker:
  '''
    Using a class method to create an instance of the PoseLandmarker - where this will provide method in order to interact with the pose pipeline
    Then w.r.t with ... as ... the instance will be automatically freed from memory after the block of code is executed.
  '''
  feature_list = normalize_and_extract_features("Images",landmarker)

with open('output.csv','w',newline='') as file:
  writer=csv.writer(file)
  for img_path, data in feature_list:
    row = [img_path] + data.tolist()
    writer.writerow(row)