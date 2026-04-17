from flask import Flask, request, jsonify
from src.search_engines.NearestNeighbours import SklearnSearchEngine
from src.predictor.predictor import predict_pose
from sklearn.neighbors import NearestNeighbors
from src.dataset.load_csv import PoseClass
import numpy as np
from src.livepipeline.normalising_coords import normalize_live_coords

app = Flask(__name__)

search_engine = SklearnSearchEngine(model=NearestNeighbors(metric="euclidean"), k=3)

pose_db = PoseClass(search_engine)
pose_db.load_csv("./dataset/pose_dataset.csv")

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.json
    raw_landmarks = np.array(data["landmarks"], dtype=np.float64)
    
    normalised_vector = normalize_live_coords(raw_landmarks)
    
    label = predict_pose(normalised_vector, pose_db)

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
