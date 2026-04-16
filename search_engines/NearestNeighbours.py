from typing import override
from interface import SearchInterface

class SklearnSearchEngine(SearchInterface):
    def __init__(self, model, k=1):
        self.model = model
        self.k = k

    @override
    def fit(self, feature_vectors):
        self.model.fit(feature_vectors)

    @override
    def search(self, unseen_vector):
        '''
            Returning a list, as kneighbours returns a list where its elements are lists (i.e. nested lists)
            Better like this, so that when it comes to having a larger k it doesnt break it.
        '''
        distances, indices = self.model.kneighbors(unseen_vector.reshape(1, -1), n_neighbors=self.k)
        return distances[0], indices[0] 