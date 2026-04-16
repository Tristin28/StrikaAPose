from flask import Flask, request, jsonify
from NearestNeighbours import SklearnSearchEngine
from predictor import predict_pose
from sklearn.neighbors import NearestNeighbors
from load_csv import PoseClass
import numpy as np


app = Flask(__name__)

search_engine = SklearnSearchEngine(model=NearestNeighbors(), k=3)

pose_db = PoseClass(search_engine)
pose_db.load_csv()

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.json
    unseen_vector = np.array(data["features"], dtype=np.float64)

    label = predict_pose(unseen_vector, pose_db)

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
