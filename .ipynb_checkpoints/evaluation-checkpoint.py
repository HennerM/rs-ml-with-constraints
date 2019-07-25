import math
import multiprocessing
from functools import lru_cache
from typing import List

from utils.common import load_dataset, printProgressBar
import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf

def disc(k):
    return 1 / np.log2(k + 1)

# Class is not thread safe!
class Evaluation:

    def __init__(self, dataset: dict):
        self.dataset = dataset
        loaded_features = np.load(dataset['item_features'], allow_pickle=True)
        self.item_features = loaded_features['features']
        self.known_frequencies = loaded_features['known_frequency']
        nr_processes = multiprocessing.cpu_count() - 2
        self.input_q = multiprocessing.JoinableQueue()
        self.output_q = multiprocessing.JoinableQueue()
        self.processes = [multiprocessing.Process(target=work_on_batch, args=(self.input_q, self.output_q, self)) for i in range(nr_processes)]
        for i in range(nr_processes):
            self.processes[i].start()


    def __del__(self):
        for p in self.processes:
            self.input_q.put(None)
            # p.terminate()

    @staticmethod
    def calculate_MSE(predictions, actual, mask):
        nr_ratings = len(mask.nonzero()[0])
        error = np.sum(np.square(predictions - actual) * mask)
        return error / nr_ratings

    @staticmethod
    def tf_calculate_accuracy(predictions, actual, mask):
        #   (tp + tn) / (tp + tn + fp + fn)
        pred = predictions > 0.5
        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(pred, tf.math.logical_and(actual, mask)), tf.int32))
        tn =  tf.reduce_sum(tf.cast((~pred) & (~actual) & mask, tf.int32))
        return (tp + tn) / tf.reduce_sum(tf.cast(mask, tf.int32))

    
    @staticmethod
    def calculate_accuracy(predictions, actual, mask):
        #   (tp + tn) / (tp + tn + fp + fn)
        pred = predictions > 0.5
        tp = (pred * actual * mask).sum()
        tn = ((~pred) * (~actual) * mask).sum()
        return (tp + tn) / mask.sum()

    @staticmethod
    def calculate_precision(predictions, actual, mask):
        #   tp / (tp + fp)
        pred = predictions > 0.5
        tp = (pred * actual * mask).sum()
        fp = (pred * (~actual) * mask).sum()
        if tp + fp == 0:
            return 0
        else:
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
    def recommend_top_n(predictions, n, without):
        sorted = np.flip(np.argsort(predictions * ~without, axis=1), axis=1)
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
        epd_values = list()

        for u in range(nr_users):
            item_set = actual[u].nonzero()[0]
            nr_items_considered = min(len(item_set), 30)
            if nr_items_considered > 0:
                item_set = np.random.choice(item_set, nr_items_considered, replace=False)
                running_sum = 0
                for i in range(nr_recs):
                    for j in range(nr_items_considered):
                        running_sum += 1 - self.item_sim(recommendations[u, i], item_set[j])
                epd_values.append(running_sum / (nr_recs * nr_items_considered))


        return sum(epd_values) / len(epd_values)

    @staticmethod
    def precision_at_n(recommendations, x_test, n):
        return np.sum(x_test[recommendations]) / n

    @staticmethod
    def average_precision(recommendations, x_test, n):
        running_sum = 0
        if np.sum(x_test) == 0:
            return 0
        for k in range(1, n + 1):
            relevant = x_test[recommendations[k - 1]]
            prec = Evaluation.precision_at_n(recommendations[:k], x_test, k)
            running_sum += prec * relevant
        return running_sum / min(n, np.sum(x_test))

    @staticmethod
    def mean_average_precision(recommendations, x_test, n):
        ap = np.zeros(recommendations.shape[0])
        for u in range(recommendations.shape[0]):
            ap[u] = Evaluation.average_precision(recommendations[u], x_test[u], n)

        return np.mean(ap)

    def calc_recommendation_metrics(self, predictions, batch):
        x = batch['x'].numpy()
        x_test = batch['x_test'].numpy()
        mask_test = batch['mask_test'].numpy()

        top_5 = Evaluation.recommend_top_n(predictions, 5, x)
        top_10 = Evaluation.recommend_top_n(predictions, 10, x)

        metrics = dict()
        metrics['accuracy'] = Evaluation.calculate_accuracy(predictions, x_test, mask_test)
        metrics['precision'] = Evaluation.calculate_precision(predictions, x_test, mask_test)
        metrics['recall'] = Evaluation.calculate_recall(predictions, x_test, mask_test)
        metrics['map@1'] = Evaluation.mean_average_precision(top_5, x_test, 1)
        metrics['map@5'] = Evaluation.mean_average_precision(top_5, x_test, 5)
        metrics['map@10'] = Evaluation.mean_average_precision(top_10, x_test, 10)
        # metrics['mse'] = Evaluation.calculate_MSE(predictions, x_test, mask_test)
        metrics['diversity@5'] = self.diversity_for_list(top_5).mean()
        metrics['diversity@10'] = self.diversity_for_list(top_10).mean()
        metrics['epc@5'] = self.expected_popularity_complement(top_5).mean()
        metrics['epc@10'] = self.expected_popularity_complement(top_10).mean()
        metrics['epd@5'] = self.expected_profile_distance(top_5, x_test)

        return metrics

    def evaluate_single_thread(self, model, mode = 'test', max_nr_batches = None):
        metrics = dict()
        nr_batches = 0
        dataset = load_dataset(self.dataset, mode).batch(512)
        if max_nr_batches is not None:
            dataset = dataset.take(max_nr_batches)
        
        for batch in dataset:
            x = batch['x']
            user_ids = batch['user_id']
            predictions = model.predict(x, user_ids)
            print('Batch nr {} predicted'.format(nr_batches + 1))

            result = self.calc_recommendation_metrics(predictions, batch)
            nr_batches += 1
            for k in result:
                metrics.setdefault(k, 0)
                metrics[k] += result[k]

        metrics = {key: (v / nr_batches) for (key, v) in metrics.items()}
        metrics['name'] = model.get_name()
        for k, v in model.get_params().items():
            metrics[k] = v

        return metrics

    def evaluate(self, model, mode = 'test', max_nr_batches = None):
        nr_batches = 0
        dataset = load_dataset(self.dataset, mode).batch(128)

        if max_nr_batches is not None:
            dataset = dataset.take(max_nr_batches)

        metrics = dict()

        for batch in dataset:
            x = batch['x']
            user_ids = batch['user_id']
            predictions = model.predict(x, user_ids)
            print('Batch nr {} predicted'.format(nr_batches + 1))
            self.input_q.put((batch, predictions, nr_batches))
            nr_batches += 1

        print('waiting for queue')
        self.input_q.join()
        self.output_q.put(None)


        print('processing results')
        i = 0
        while True:
            result = self.output_q.get()
            if result is None:
                break
            i += 1
            printProgressBar(i, nr_batches, 'Evaluating {}'.format(model.get_name()), length=60)


            for k in result:
                metrics.setdefault(k, 0)
                metrics[k] += result[k]
            self.output_q.task_done()

        metrics = {key: (v / nr_batches) for (key, v) in metrics.items()}
        metrics['name'] = model.get_name()
        for k, v in model.get_params().items():
            metrics[k] = v

        return metrics

    def evaluate_models(self, model: list):
        values = [self.evaluate(m) for m in model]
        return pd.DataFrame(values)


def work_on_batch(input_queue, output_queue, evaluation):
    while True:
        data = input_queue.get()
        print("Process batch")
        start = time.time()
        if data is None:
            input_queue.task_done()
            break

        batch, predictions, batch_nr = data

        output = evaluation.calc_recommendation_metrics(predictions, batch)
        print("Processing batch done")
        output_queue.put(output)
        input_queue.task_done()


#
# if __name__ == "__main__":
#     ev = Evaluation(movie_lens)
#     model = ConstraintAutoRec(movie_lens['dimensions'], epochs=5)
#     # model.train(load_dataset(movie_lens, 'train'), movie_lens['train']['records'])
#     ev.evaluate(model)

