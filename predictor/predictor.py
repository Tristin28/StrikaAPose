from dataset.load_csv import PoseClass
from collections import Counter

def predict_pose(unseen_vector, pose_store: PoseClass):
    if not pose_store.loaded():
        raise RuntimeError("Pose data not loaded into RAM")
    
    search_engine = pose_store.get_search_engine()
    labels = pose_store.get_labels()

    _, indices = search_engine.search(unseen_vector)
    
    neighbour_labels = [labels[i] for i in indices]
    label_dict = Counter(neighbour_labels)

    predicted_label = label_dict.most_common(1)[0][0]

    return predicted_label