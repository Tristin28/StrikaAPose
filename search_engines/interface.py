'''
    Created an interface for the reason that if other search engine algorithms are introduced they will follow the same contract design (API calls)
    Which is useful as when dataset scales more then we would need more efficient algorithms which perform better than linear time, i.e. O(n)
'''
from abc import ABC, abstractmethod

class SearchInterface(ABC):
    @abstractmethod
    def search(self,unseen_vector):
        pass

    @abstractmethod
    def fit(self,feature_vector):
        pass