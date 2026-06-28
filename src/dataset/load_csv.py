import pandas as pd
import numpy as np

class PoseClass:
    def __init__(self, search_engines):
        '''
            search_engines is a dict mapping a metric name to a search engine, e.g. {"euclidean": SklearnSearchEngine(...), "cosine": SklearnSearchEngine(...)} 
            so the user can compare different similarity metrics from the UI.
        '''
        self.labels = None
        self.features = None
        self.image_paths = None
        self.search_engines = search_engines
        self.is_loaded = False

    def load_csv(self,csv_path):
        df = pd.read_csv(csv_path)

        self.image_paths = df["image_path"].to_numpy() #path of the original image so we can show the closest match
        self.labels = df["label"].to_numpy() #Converting to a numpy data strucutre rather than pandas because scikit-learn algo. expects this structure
        self.features = df.drop(columns=["image_path", "label"]).to_numpy(dtype=np.float64)

        for search_engine in self.search_engines.values():
            search_engine.fit(self.features) #same features, fit once per metric

        self.is_loaded = True

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_image_paths(self):
        return self.image_paths

    def loaded(self):
        return self.is_loaded

    def get_search_engine(self, metric="euclidean"):
        return self.search_engines[metric]
