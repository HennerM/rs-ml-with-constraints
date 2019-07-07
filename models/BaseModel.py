from abc import abstractmethod
import tensorflow as tf
import numpy as np

class BaseModel:
    def __init__(self, dimensions, **kwargs):
        self.dimensions = dimensions
        self.args = kwargs


    @abstractmethod
    def train(self, dataset: tf.data.Dataset, nr_records: int):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_params(self) -> dict:
        pass
