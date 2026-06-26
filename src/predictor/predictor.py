from collections import Counter

from src.dataset.load_csv import PoseClass

# Open-set guard: nearest neighbours always return something, even for a neutral
# or unrelated pose. These limits decide when the closest result is still too far
# away to be treated as a real match. Tune these if the dataset changes a lot.
MATCH_DISTANCE_LIMITS = {
    "euclidean": 8.5,
    "cosine": 0.055,
    "manhattan": 55.0,
}

# If the live pose is slightly over the strict distance limit, a strong top-5
# neighbour vote can still accept it. This helps live-camera matching, where the
# pose is noisier than the cleaned dataset images.
STRONG_VOTE_MIN_NEIGHBOURS = 4
STRONG_VOTE_DISTANCE_LIMITS = {
    "euclidean": 10.5,
    "cosine": 0.075,
    "manhattan": 70.0,
}

# With 5 neighbours, 3 votes is a real majority. This lets the prediction ignore
# one unusually close outlier when most neighbours agree on another label.
MAJORITY_VOTE_MIN_NEIGHBOURS = 3

# The closest neighbour also needs to be clearly better than the nearest
# different-label neighbour. Similar same-label neighbours are useful evidence,
# not ambiguity.
MIN_DISTANCE_MARGINS = {
    "euclidean": 1.0,
    "cosine": 0.018,
    "manhattan": 10.0,
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

    distance_limit = MATCH_DISTANCE_LIMITS.get(metric)
    strong_vote_distance_limit = STRONG_VOTE_DISTANCE_LIMITS.get(metric)
    minimum_margin = MIN_DISTANCE_MARGINS.get(metric)
    label_votes = Counter(neighbour["label"] for neighbour in neighbours)
    majority_label, majority_vote_count = label_votes.most_common(1)[0]
    neighbour_count = len(neighbours)
    majority_vote = majority_vote_count >= MAJORITY_VOTE_MIN_NEIGHBOURS

    if majority_vote:
        predicted_label = str(majority_label)
        best_match = next(neighbour for neighbour in neighbours if neighbour["label"] == predicted_label)
        predicted_vote_count = int(majority_vote_count)
    else:
        best_match = neighbours[0]
        predicted_label = best_match["label"]
        predicted_vote_count = int(label_votes[predicted_label])

    nearest_competing_match = next((neighbour for neighbour in neighbours if neighbour["label"] != predicted_label), None)
    distance_margin = None
    if nearest_competing_match is not None:
        distance_margin = float(nearest_competing_match["distance"] - best_match["distance"])

    if not majority_vote and minimum_margin is not None and distance_margin is not None and distance_margin < minimum_margin:
        return {
            "prediction": None,
            "metric": metric,
            "match_found": False,
            "message": "No confident pose match. The nearest matches are too similar.",
            "best_match": best_match,
            "neighbours": neighbours,
            "nearest_competing_match": nearest_competing_match,
            "distance_margin": distance_margin,
            "minimum_margin": minimum_margin,
            "predicted_vote_count": predicted_vote_count,
            "neighbour_count": neighbour_count,
        }

    strong_neighbour_vote = (
        predicted_vote_count >= STRONG_VOTE_MIN_NEIGHBOURS
        and strong_vote_distance_limit is not None
        and best_match["distance"] <= strong_vote_distance_limit
    )

    if distance_limit is not None and best_match["distance"] > distance_limit and not strong_neighbour_vote:
        return {
            "prediction": None,
            "metric": metric,
            "match_found": False,
            "message": "No confident pose match. The closest pose is still too far away.",
            "best_match": best_match,
            "neighbours": neighbours,
            "threshold": distance_limit,
            "strong_vote_threshold": strong_vote_distance_limit,
            "predicted_vote_count": predicted_vote_count,
            "neighbour_count": neighbour_count,
        }

    return {
        "prediction": predicted_label,
        "metric": metric,
        "match_found": True,
        "best_match": best_match, #the single closest image, which is what we display
        "neighbours": neighbours,
        "accepted_by": (
            "strong_neighbour_vote"
            if strong_neighbour_vote
            else "majority_vote"
            if majority_vote
            else "strict_distance"
        ),
        "predicted_vote_count": predicted_vote_count,
        "neighbour_count": neighbour_count,
    }
