import math
import multiprocessing
from functools import lru_cache
from pathos.multiprocessing import ProcessingPool as Pool
from typing import List

from models import BaseModel
from models.ConstraintAutoRec import ConstraintAutoRec
from utils.common import movie_lens, load_dataset, load_testset
import numpy as np
import pandas as pd
import time

def disc(k):
    return 1 / np.log2(k + 1)


class Evaluation:

    def __init__(self, dataset: dict):
        self.dataset = dataset
        loaded_features = np.load(dataset['item_features'], allow_pickle=True)
        self.item_features = loaded_features['features']
        self.known_frequencies = loaded_features['known_frequency']

    @staticmethod
    def calculate_MSE(predictions, actual, with_held):
        error = np.square(predictions - actual)
        total_error = 0
        for i in range(error.shape[0]):
            with_held_items = with_held[i].nonzero()[0]
            total_error += error[i, with_held_items].mean()

        return total_error / float(error.shape[0])

    @staticmethod
    def calculate_accuracy(predictions, actual, mask):
        #   tp / (tp + fp)
        pred = predictions > 0.5
        tp = (pred * actual * mask).sum()
        fp = (pred * (~actual) * mask).sum()
        return (tp + fp) / mask.sum()

    @staticmethod
    def calculate_precision(predictions, actual, mask):
        #   tp / (tp + fp)
        pred = predictions > 0.5
        tp = (pred * actual * mask).sum()
        fp = (pred * (~actual) * mask).sum()
        return tp / (tp + fp)

    @staticmethod
    def calculate_recall(predictions, actual, mask):
        #   tp / (tp + fn)
        pred = predictions > 0.5
        tp = (pred * actual * mask).sum()
        fn = ((~pred) * actual * mask).sum()
        return tp / (tp + fn)

    @lru_cache(maxsize=4096)
    def item_sim(self, item_id, other_id):
        return Evaluation.cosine_similarity(self.item_features[item_id], self.item_features[other_id])


    @staticmethod
    def cosine_similarity(u, v):
        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            return 0
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    @staticmethod
    def recommend_top_n(predictions, n):
        sorted = np.flip(np.argsort(predictions, axis=1), axis=1)
        return sorted[:, 0:n]


    def calc_diversity(self, items):
        n = len(items)
        running_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                running_sum += 1 - self.item_sim(items[i], items[j])

        return running_sum / ((n-1) * (n/2))


    def diversity_for_list(self, recs):
        return np.apply_along_axis(lambda row: self.calc_diversity(row), 1, recs)


    def expected_popularity_complement(self, recommendations):
        def epc(recs):
            discounts = disc(np.arange(1, len(recs) + 1))
            freqs = 1 - self.known_frequencies[recs]
            return np.sum(discounts * freqs.T) / np.sum(discounts)

        return np.apply_along_axis(epc, 1, recommendations)

    def expected_profile_distance(self, recommendations, actual):
        nr_users = recommendations.shape[0]
        nr_recs = recommendations.shape[1]
        epd_values = np.zeros(nr_users)

        for u in range(nr_users):
            item_set = actual[u].nonzero()[0]
            nr_items_considered = min(len(item_set), 30)
            item_set = np.random.choice(item_set, nr_items_considered, replace=False)
            running_sum = 0
            for i in range(nr_recs):
                for j in range(nr_items_considered):
                    running_sum += 1 - self.item_sim(recommendations[u, i], item_set[j])
            epd_values[u] = running_sum / (nr_recs * nr_items_considered)


        return epd_values


    def calc_batch_metrics(self, model, batch):
        x = batch['x'].numpy()
        y = batch['y'].numpy()
        mask = batch['mask'].numpy()
        with_held = batch['held_back'].numpy()

        batch_start = time.time()

        predictions = model.predict(x)
        top_5 = Evaluation.recommend_top_n(predictions, 5)
        top_10 = Evaluation.recommend_top_n(predictions, 10)

        metrics = dict()
        metrics['accuracy'] = Evaluation.calculate_accuracy(predictions, y, mask)
        metrics['precision'] = Evaluation.calculate_precision(predictions, y, mask)
        metrics['recall'] = Evaluation.calculate_recall(predictions, y, mask)
        metrics['mse'] = Evaluation.calculate_MSE(predictions, y, with_held)
        metrics['diversity@5'] = self.diversity_for_list(top_5).mean()
        metrics['diversity@10'] = self.diversity_for_list(top_10).mean()
        metrics['epc@5'] = self.expected_popularity_complement(top_5).mean()
        metrics['epc@10'] = self.expected_popularity_complement(top_10).mean()
        metrics['epd@5'] = self.expected_profile_distance(top_5, y).mean()

        return metrics


    def evaluate(self, model: BaseModel):
        def calc_batch(batch):
            return self.calc_batch_metrics(model, batch)

        nr_batches = 0
        dataset = load_testset(self.dataset).batch(128)

        multiprocessing.Process()
        metrics = dict()

        for batch in dataset:
            print('Process batch', nr_batches)

            m = self.calc_batch_metrics(model, batch)
            for k in m:
                metrics.setdefault(k, 0)
                metrics[k] += m[k]
            nr_batches += 1

        metrics = {key: (v / nr_batches) for (key, v) in metrics.items()}

        metrics['name'] = model.get_name()
        for k, v in model.get_params().items():
            metrics[k] = v

        return metrics

    def evaluate_models(self, model: list):
        values = [self.evaluate(m) for m in model]
        return pd.DataFrame(values)



if __name__ == "__main__":
    model = ConstraintAutoRec(movie_lens['dimensions'])
    model.train(load_dataset(movie_lens, 'train'), movie_lens['train']['records'])

