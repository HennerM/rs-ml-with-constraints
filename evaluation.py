import math
from typing import List

from models import BaseModel
from models.ConstraintAutoRec import ConstraintAutoRec
from utils.common import movie_lens, load_dataset, load_testset
import numpy as np
import pandas as pd

def calculate_MSE(predictions, actual, with_held):
    error = np.square(predictions - actual)
    total_error = 0
    for i in range(error.shape[0]):
        with_held_items = with_held[i].nonzero()[0]
        total_error += error[i, with_held_items].mean()

    return total_error / float(error.shape[0])

def calculate_accuracy(predictions, actual, mask):
    #   tp / (tp + fp)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fp = (pred * (~actual) * mask).sum()
    return (tp + fp) / mask.sum()


def calculate_precision(predictions, actual, mask):
    #   tp / (tp + fp)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fp = (pred * (~actual) * mask).sum()
    return tp / (tp + fp)

def calculate_recall(predictions, actual, mask):
    #   tp / (tp + fn)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fn = ((~pred) * actual * mask).sum()
    return tp / (tp + fn)

def cosine_similarity(u, v):
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 0
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def recommend_top_n(predictions, n):
    sorted = np.flip(np.argsort(predictions, axis=1), axis=1)
    return sorted[:, 0:n]



def calc_diversity_with_features(item_features):
    def calc_diversity(items):
        n = len(items)
        running_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                running_sum += 1 - cosine_similarity(item_features[items[i]], item_features[items[j]])

        return running_sum / ((n-1) * (n/2))
    return calc_diversity

def diversity_for_list(recs, item_features):
    return np.apply_along_axis(calc_diversity_with_features(item_features), 1, recs)


def disc(k):
    return 1 / np.log2(k + 1)

def expected_popularity_complement(recommendations, known_frequencies):
    def epc(recs):
        discounts = disc(np.arange(1, len(recs) + 1))
        freqs = 1 - known_frequencies[recs]
        return np.sum(discounts * freqs.T) / np.sum(discounts)

    return np.apply_along_axis(epc, 1, recommendations)

def expected_profile_distance(recommendations, actual, item_features):
    nr_users = recommendations.shape[0]
    nr_recs = recommendations.shape[1]
    epd_values = np.zeros(nr_users)

    for u in range(nr_users):
        item_set = actual[u].nonzero()[0]
        running_sum = 0
        for i in range(nr_recs):
            for j in range(len(item_set)):
                running_sum += 1 - cosine_similarity(item_features[recommendations[u, i]], item_features[item_set[j]])
        epd_values[u] = running_sum / (nr_recs * len(item_set))


    return epd_values


def evaluate(model: BaseModel, dataset: dict):

    loaded_features = np.load(dataset['item_features'], allow_pickle=True)
    item_features = loaded_features['features']
    known_frequencies = loaded_features['known_frequency']

    dataset = load_testset(dataset).batch(32)

    metrics = dict()
    for m in ['accuracy', 'precision', 'recall', 'mse', 'diversity@5', 'diversity@10', 'epc@5', 'epc@10', 'epd@5']:
        metrics[m] = 0
    nr_batches = 0

    for batch in dataset:
        x = batch['x'].numpy()
        y = batch['y'].numpy()
        mask = batch['mask'].numpy()
        with_held = batch['held_back'].numpy()

        predictions = model.predict(x)
        top_5 = recommend_top_n(predictions, 5)
        top_10 = recommend_top_n(predictions, 10)

        metrics['accuracy'] += calculate_accuracy(predictions, y, mask)
        metrics['precision'] += calculate_precision(predictions, y, mask)
        metrics['recall'] += calculate_recall(predictions, y, mask)
        metrics['mse'] += calculate_MSE(predictions, y, with_held)

        metrics['diversity@5'] = +diversity_for_list(top_5, item_features).mean()
        metrics['diversity@10'] += diversity_for_list(top_10, item_features).mean()

        metrics['epc@5'] += expected_popularity_complement(top_5, known_frequencies).mean()
        metrics['epc@10'] += expected_popularity_complement(top_10, known_frequencies).mean()

        # metrics['epd@5'] =+ expected_profile_distance(top_5, y, item_features).mean()

        nr_batches += 1

    metrics = {key: (v / nr_batches) for (key, v) in metrics.items()}

    metrics['name'] = model.get_name()
    for k, v in model.get_params().items():
        metrics[k] = v

    return metrics

def evaluate_models(model: list, dataset: dict):
    values = [evaluate(m, dataset) for m in model]
    return pd.DataFrame(values)



if __name__ == "__main__":
    model = ConstraintAutoRec(movie_lens['dimensions'])
    model.train(load_dataset(movie_lens, 'train'), movie_lens['train']['records'])

