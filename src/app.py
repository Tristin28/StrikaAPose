from flask import Flask, request, jsonify, render_template, send_from_directory
from src.search_engines.NearestNeighbours import SklearnSearchEngine
from src.predictor.predictor import predict_pose
from sklearn.neighbors import NearestNeighbors
from src.dataset.load_csv import PoseClass
from src.livepipeline.normalising_coords import normalize_live_coords
from src.datapipeline.config import DATASET_PATH, PROJECT_ROOT

#Serving the frontend through Flask (templates + static) so there are no cross-origin (CORS) problems.
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

#One fitted search engine per similarity metric, so the UI can switch between them.
METRICS = ["euclidean", "cosine", "manhattan"]
search_engines = {metric: SklearnSearchEngine(model=NearestNeighbors(metric=metric), k=3) for metric in METRICS}

pose_db = PoseClass(search_engines)
pose_db.load_csv(DATASET_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Images/<path:filename>")
def dataset_image(filename):
    #lets the frontend load the closest matching image from the dataset
    return send_from_directory(str(PROJECT_ROOT / "Images"), filename)

@app.route("/Model/<path:filename>")
def model_file(filename):
    #serves the MediaPipe model so the browser can load the same model the dataset was built with
    return send_from_directory(str(PROJECT_ROOT / "Model"), filename)

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.json
    raw_landmarks = data["landmarks"]
    metric = data.get("metric", "euclidean")
    if metric not in search_engines:
        metric = "euclidean"

    normalised_vector = normalize_live_coords(raw_landmarks)
    if normalised_vector is None:
        return jsonify({"error": "Invalid pose input"}), 400

    result = predict_pose(normalised_vector, pose_db, metric)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
