from abc import abstractmethod
import tensorflow as tf

class BaseModel:
    def __init__(self, nr_users, nr_items, **kwargs):
        self.nr_users = nr_users
        self.nr_items = nr_items
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
    def test(self, input_data):
        pass


    def recommend(self, input_data):
        pass
