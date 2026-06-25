from src.dataset.load_csv import PoseClass
from collections import Counter

def predict_pose(unseen_vector, pose_store: PoseClass, metric="euclidean"):
    if not pose_store.loaded():
        raise RuntimeError("Pose data not loaded into RAM")

    search_engine = pose_store.get_search_engine(metric)
    labels = pose_store.get_labels()
    image_paths = pose_store.get_image_paths()

    distances, indices = search_engine.search(unseen_vector)

    neighbours = []
    for distance, index in zip(distances, indices):
        neighbours.append({
            "label": str(labels[index]),
            "distance": float(distance), #cast away numpy types so jsonify can serialise them
            "image": str(image_paths[index])
        })

    #majority vote across the k neighbours decides the predicted class
    neighbour_labels = [neighbour["label"] for neighbour in neighbours]
    predicted_label = Counter(neighbour_labels).most_common(1)[0][0]

    return {
        "prediction": predicted_label,
        "metric": metric,
        "best_match": neighbours[0] #the single closest image, which is what we display
    }