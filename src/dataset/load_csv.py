import pandas as pd
import numpy as np

class PoseClass:
    def __init__(self, search_engine):
        self.labels = None
        self.features = None
        self.search_engine = search_engine
        self.is_loaded = False

    def load_csv(self,csv_path):
        df = pd.read_csv(csv_path)

        self.labels = df["label"].to_numpy() #Converting to a numpy data strucutre rather than pandas because scikit-learn algo. expects this structure
        self.features = df.drop(columns=["label"]).to_numpy(dtype=np.float64)

        self.search_engine.fit(self.features)
        
        self.is_loaded = True
    
    def get_features(self):
        return self.features
    
    def get_labels(self):
        return self.labels
    
    def loaded(self):
        return self.is_loaded
    
    def get_search_engine(self):
        return self.search_engine
