from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.datapipeline.config import DATASET_PATH


METRICS = ("euclidean", "cosine", "manhattan")
SCALING_MODES = ("raw", "standardized")

@dataclass
class MetricResult:
    metric: str
    scaling: str
    total: int
    top_1_correct: int
    top_k_correct: int
    avg_correct_distance: float | None
    avg_wrong_distance: float | None
    confusion: Counter

    @property
    def top_1_accuracy(self):
        return self.top_1_correct / self.total if self.total else 0.0

    @property
    def top_k_accuracy(self):
        return self.top_k_correct / self.total if self.total else 0.0


def majority_vote(neighbour_labels):
    counts = Counter(neighbour_labels)
    winning_count = max(counts.values())

    # Resolve ties by nearest neighbour order.
    for label in neighbour_labels:
        if counts[label] == winning_count:
            return label

    raise RuntimeError("Could not compute majority vote")


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = df["label"].to_numpy()
    features = df.drop(columns=["image_path", "label"]).to_numpy(dtype=np.float64)
    return features, labels


def remove_singleton_labels(features, labels):
    label_counts = Counter(labels)
    excluded = {label: count for label, count in label_counts.items() if count < 2}
    keep_mask = np.array([label_counts[label] >= 2 for label in labels])
    return features[keep_mask], labels[keep_mask], excluded


def choose_fold_count(labels, requested_folds):
    '''

    '''
    min_label_count = min(Counter(labels).values())
    return min(requested_folds, min_label_count)


def evaluate_metric(features, labels, metric, folds, k, random_state, scaling):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    top_1_correct = 0
    top_k_correct = 0
    total = 0
    correct_distances = []
    wrong_distances = []
    confusion = Counter()

    for train_indices, test_indices in splitter.split(features, labels):
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        test_features = features[test_indices]
        test_labels = labels[test_indices]

        if scaling == "standardized":
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)

        neighbour_count = min(k, len(train_features))
        model = NearestNeighbors(metric=metric, n_neighbors=neighbour_count)
        model.fit(train_features)

        distances, neighbour_indices = model.kneighbors(test_features)

        for row_index, true_label in enumerate(test_labels):
            matched_labels = train_labels[neighbour_indices[row_index]]
            predicted_label = majority_vote(matched_labels)
            best_distance = float(distances[row_index][0])

            total += 1
            if predicted_label == true_label:
                top_1_correct += 1
                correct_distances.append(best_distance)
            else:
                wrong_distances.append(best_distance)
                confusion[(true_label, predicted_label)] += 1

            if true_label in matched_labels:
                top_k_correct += 1

    return MetricResult(metric=metric, scaling=scaling, total=total, top_1_correct=top_1_correct, top_k_correct=top_k_correct,
                        avg_correct_distance=np.mean(correct_distances).item() if correct_distances else None,
                        avg_wrong_distance=np.mean(wrong_distances).item() if wrong_distances else None, confusion=confusion)


'''
    Helper methods to help in displaying the results.
'''
def format_percent(value):
    return f"{value * 100:6.2f}%"

def format_distance(value):
    return "n/a" if value is None else f"{value:.4f}"

def print_results(results, k):
    print("\nMetric Evaluation Results")
    print("-" * 110)
    print(
        f"{'Scaling':<16}"
        f"{'Metric':<12}"
        f"{'Top-1 Accuracy':>18}"
        f"{f'Top-{k} Accuracy':>18}"
        f"{'Avg Correct Dist':>22}"
        f"{'Avg Wrong Dist':>20}"
    )
    print("-" * 110)

    for result in results:
        print(
            f"{result.scaling:<16}"
            f"{result.metric:<12}"
            f"{format_percent(result.top_1_accuracy):>18}"
            f"{format_percent(result.top_k_accuracy):>18}"
            f"{format_distance(result.avg_correct_distance):>22}"
            f"{format_distance(result.avg_wrong_distance):>20}"
        )

    best = max(results, key=lambda result: (result.top_1_accuracy, result.top_k_accuracy))
    print("-" * 110)
    print(f"Best setup by Top-1 accuracy: {best.metric} with {best.scaling} features")

    for result in results:
        if not result.confusion:
            continue

        print(f"\nMost common mistakes for {result.metric} with {result.scaling} features:")
        for (true_label, predicted_label), count in result.confusion.most_common(5):
            print(f"  {true_label} predicted as {predicted_label}: {count}")


def parse_args():
    parser = ArgumentParser(description="Evaluate pose similarity metrics with stratified cross-validation.")
    parser.add_argument("--csv", default=str(DATASET_PATH), help="Path to pose_dataset.csv")
    parser.add_argument("--folds", type=int, default=5, help="Requested number of stratified folds")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbours used for voting")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for fold shuffling")
    parser.add_argument(
        "--scaling",
        choices=("raw", "standardized", "both"),
        default="raw",
        help="Feature scaling mode. Use 'both' to compare raw and standardized features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features, labels = load_dataset(args.csv)
    original_count = len(labels)
    features, labels, excluded = remove_singleton_labels(features, labels)

    if len(labels) < 2:
        raise ValueError("Not enough evaluable rows. Add at least two images for at least two labels.")

    fold_count = choose_fold_count(labels, args.folds)
    if fold_count < 2:
        raise ValueError("Cross-validation needs at least two folds. Add more images per label.")

    print("Cross-validation setup")
    print(f"Dataset: {args.csv}")
    print(f"Rows loaded: {original_count}")
    print(f"Rows evaluated: {len(labels)}")
    print(f"Requested folds: {args.folds}")
    print(f"Actual folds: {fold_count}")
    print(f"Neighbours per prediction: {args.k}")
    print(f"Feature scaling: {args.scaling}")

    if excluded:
        print("\nExcluded labels with fewer than 2 examples:")
        for label, count in sorted(excluded.items()):
            print(f"  {label}: {count}")

    scaling_modes = SCALING_MODES if args.scaling == "both" else (args.scaling,)
    results = [
        evaluate_metric(features, labels, metric, fold_count, args.k, args.random_state, scaling)
        for scaling in scaling_modes
        for metric in METRICS
    ]
    print_results(results, args.k)


if __name__ == "__main__":
    main()
