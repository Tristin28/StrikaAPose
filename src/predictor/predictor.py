from src.dataset.load_csv import PoseClass

# Open-set guard: nearest neighbours always return something, even for a neutral
# or unrelated pose. These limits decide when the closest result is still too far
# away to be treated as a real match. Tune these if the dataset changes a lot.
MATCH_DISTANCE_LIMITS = {
    "euclidean": 9.5,
    "cosine": 0.07,
    "manhattan": 75.0,
}

# The closest neighbour also needs to be clearly better than the nearest
# different-label neighbour. Similar same-label neighbours are useful evidence,
# not ambiguity.
MIN_DISTANCE_MARGINS = {
    "euclidean": 0.75,
    "cosine": 0.012,
    "manhattan": 8.0,
}

def predict_pose(unseen_vector, pose_store: PoseClass, metric="cosine"):
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

    best_match = neighbours[0]
    predicted_label = best_match["label"]
    distance_limit = MATCH_DISTANCE_LIMITS.get(metric)
    minimum_margin = MIN_DISTANCE_MARGINS.get(metric)

    if distance_limit is not None and best_match["distance"] > distance_limit:
        return {
            "prediction": None,
            "metric": metric,
            "match_found": False,
            "message": "No confident pose match. The closest pose is still too far away.",
            "best_match": best_match,
            "threshold": distance_limit
        }

    nearest_competing_match = next((neighbour for neighbour in neighbours[1:] if neighbour["label"] != predicted_label), None)
    if minimum_margin is not None and nearest_competing_match is not None:
        distance_margin = nearest_competing_match["distance"] - best_match["distance"]
        if distance_margin < minimum_margin:
            return {
                "prediction": None,
                "metric": metric,
                "match_found": False,
                "message": "No confident pose match. The nearest matches are too similar.",
                "best_match": best_match,
                "nearest_competing_match": nearest_competing_match,
                "distance_margin": distance_margin,
                "minimum_margin": minimum_margin
            }

    return {
        "prediction": predicted_label,
        "metric": metric,
        "match_found": True,
        "best_match": best_match #the single closest image, which is what we display
    }
