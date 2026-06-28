from threading import Timer
import webbrowser

from flask import Flask, request, jsonify, render_template, send_from_directory
from src.search_engines.NearestNeighbours import SklearnSearchEngine
from src.predictor.predictor import predict_pose
from sklearn.neighbors import NearestNeighbors
from src.dataset.load_csv import PoseClass
from src.livepipeline.normalising_coords import assess_live_pose_activity, normalize_live_pose
from src.datapipeline.config import DATASET_PATH, PROJECT_ROOT

#Serving the frontend through Flask (templates + static) so there are no cross-origin (CORS) problems.
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

'''
    One fitted search engine per similarity metric, so the UI can switch between them.
    We retrieve 5 neighbours so the predictor can reject ambiguous matches and the debug panel can show the nearest live matches. 
    The predictor uses those neighbours for majority voting and confidence checks.
'''
METRICS = ["euclidean", "cosine", "manhattan"]
search_engines = {metric: SklearnSearchEngine(model=NearestNeighbors(metric=metric), k=5) for metric in METRICS}

pose_db = PoseClass(search_engines)
pose_db.load_csv(DATASET_PATH)

VISIBILITY_THRESHOLD = 0.5
KEY_LANDMARKS_FOR_MATCH = {
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14,
    "left wrist": 15,
    "right wrist": 16,
    "left hip": 23,
    "right hip": 24,
}


def validate_landmark_visibility(visibility):
    if visibility is None:
        return None

    if not isinstance(visibility, list) or len(visibility) != 33:
        return "Invalid landmark visibility input."

    try:
        visibility_values = [float(value or 0) for value in visibility]
    except (TypeError, ValueError):
        return "Invalid landmark visibility input."

    missing_key_landmarks = [
        name
        for name, index in KEY_LANDMARKS_FOR_MATCH.items()
        if visibility_values[index] < VISIBILITY_THRESHOLD
    ]
    if missing_key_landmarks:
        missing = ", ".join(missing_key_landmarks)
        return f"Key pose points are not visible enough: {missing}."

    return None

def open_browser(url):
    webbrowser.open_new(url)

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
    metric = data.get("metric", "cosine") #Default metric just in case the frontend doesn't send one, but it should always send one.
    if metric not in search_engines:
        metric = "cosine"

    visibility_error = validate_landmark_visibility(data.get("visibility"))
    if visibility_error is not None:
        return jsonify({
            "prediction": None,
            "metric": metric,
            "match_found": False,
            "message": visibility_error
        })

    normalized_pose = normalize_live_pose(raw_landmarks)
    if normalized_pose is None:
        return jsonify({"error": "Invalid pose input"}), 400

    normalised_coords, normalised_vector = normalized_pose
    result = predict_pose(normalised_vector, pose_db, metric)
    pose_activity = assess_live_pose_activity(normalised_coords)
    if not pose_activity["active"]:
        result.update({
            "prediction": None,
            "match_found": False,
            "message": "No confident pose match. Make a clearer active pose.",
            "pose_activity": pose_activity,
        })

    return jsonify(result)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    Timer(1.0, open_browser, args=(f"http://{host}:{port}",)).start()
    app.run(host=host, port=port, debug=True, use_reloader=False)
