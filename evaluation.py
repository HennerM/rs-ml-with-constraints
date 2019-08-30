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
        self.nr_items = dataset['dimensions']
#         nr_processes = multiprocessing.cpu_count() - 2
#         self.input_q = multiprocessing.JoinableQueue()
#         self.output_q = multiprocessing.JoinableQueue()
#         self.processes = [multiprocessing.Process(target=work_on_batch, args=(self.input_q, self.output_q, self)) for i in range(nr_processes)]
#         for i in range(nr_processes):
#             self.processes[i].start()


#     def __del__(self):
#         for p in self.processes:
#             self.input_q.put(None)
            # p.terminate()

    @staticmethod
    def calculate_MSE(predictions, actual, mask):
        nr_ratings = mask.sum()
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
    def calculate_accuracy(pred, actual, mask):
        #   (tp + tn) / (tp + tn + fp + fn)
        tp = (pred * actual * mask).sum(axis=1)
        tn = ((~pred) * (~actual) * mask).sum(axis=1)
        return (tp + tn) / mask.sum(axis=1)

    @staticmethod
    def calculate_precision(pred, actual, mask):
        #   tp / (tp + fp)
        tp = (pred * actual * mask).sum(axis=1)
        fp = (pred * (~actual) * mask).sum(axis=1)
        denom = (tp + fp)
        not_null = np.where(denom)
        if len(not_null) == 0:
            return np.array([])
        return tp[not_null] / denom[not_null]

    @staticmethod
    def calculate_recall(pred, actual, mask):
        #   tp / (tp + fn)
        tp = (pred * actual * mask).sum(axis=1)
        fn = ((~pred) * actual * mask).sum(axis=1)
        denom =  (tp + fn)
        not_null = np.where(denom)
        if len(not_null) == 0:
            return np.array([])
        return tp[not_null] / denom[not_null]

    @lru_cache(maxsize=4096)
    def item_sim(self, item_id, other_id):
        return (Evaluation.cosine_similarity(self.item_features[item_id], self.item_features[other_id]) + 1) / 2


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
                np.random.seed(u)
                item_set = np.random.choice(item_set, nr_items_considered, replace=False)
                running_sum = 0
                for i in range(nr_recs):
                    distance = 0
                    for j in range(nr_items_considered):
                        distance += (1 - self.item_sim(recommendations[u, i], item_set[j]))
                    running_sum += (distance / nr_items_considered) * disc(i + 1)
                discounts = disc(np.arange(1, nr_recs + 1))
                mean = running_sum / np.sum(discounts)
                epd_values.append(mean)


        return epd_values

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

        return ap

    @staticmethod
    def precision_at(recommendations, x_test, n):
        ap = np.zeros(recommendations.shape[0])
        for u in range(recommendations.shape[0]):
            ap[u] = Evaluation.precision_at_n(recommendations[u], x_test[u], n)

        return ap

    @staticmethod
    def recall_at(recommendations, x_test, n):
        ap = np.zeros(recommendations.shape[0])
        for u in range(recommendations.shape[0]):
            ap[u] =  np.sum(x_test[u][recommendations[u]]) / np.sum(x_test[u])

        return ap

    def calc_recommendation_metrics(self, predictions, batch):
        x = batch['x'].numpy()
        x_test = batch['x_test'].numpy()
        mask_test = batch['mask_test'].numpy()
        mask = batch['mask'].numpy()

        top_10 = Evaluation.recommend_top_n(predictions, 10, mask)
        top_5 = top_10[:,0:5]
        top_1 = top_10[:,0:1]

        pred_true = predictions > 0.5
        
        metrics = dict()
        metrics['accuracy'] = Evaluation.calculate_accuracy(pred_true, x_test, mask_test).tolist()
        metrics['precision@5'] = Evaluation.precision_at(top_5, x_test, 5).tolist()
        metrics['recall@5'] = Evaluation.recall_at(top_5, x_test, 5).tolist()
        metrics['map@1'] = Evaluation.mean_average_precision(top_5, x_test, 1).tolist()
        metrics['map@5'] = Evaluation.mean_average_precision(top_5, x_test, 5).tolist()
        metrics['map@10'] = Evaluation.mean_average_precision(top_10, x_test, 10).tolist()
        metrics['diversity@5'] = self.diversity_for_list(top_5).tolist()
        metrics['diversity@10'] = self.diversity_for_list(top_10).tolist()
        metrics['epc@5'] = self.expected_popularity_complement(top_5).tolist()
        metrics['epc@10'] = self.expected_popularity_complement(top_10).tolist()
        metrics['epd@5'] = self.expected_profile_distance(top_5, x_test)
        metrics['unique@1'] = top_1
        metrics['unique@5'] = top_5
        metrics['unique@10'] = top_10
        metrics['nr_users'] = len(x)
        return metrics

    def init_metrics(self):
        metrics = dict()
        metrics['accuracy'] = list()
        metrics['precision@5'] = list()
        metrics['recall@5'] = list()
        metrics['map@1'] = list()
        metrics['map@5'] = list()
        metrics['map@10'] = list()
        metrics['diversity@5'] = list()
        metrics['diversity@10'] =list()
        metrics['epc@5'] = list()
        metrics['epc@10'] = list()
        metrics['epd@5'] = list()
        metrics['unique@1'] = np.zeros(self.nr_items)
        metrics['unique@5'] = np.zeros(self.nr_items)
        metrics['unique@10'] = np.zeros(self.nr_items)
        metrics['nr_users'] = 0.0
        return metrics
    
    @staticmethod
    def update_metrics(metrics, curr):
        metrics['accuracy'] += curr['accuracy']
        metrics['precision@5'] += curr['precision@5']
        metrics['recall@5'] += curr['recall@5']
        metrics['map@1'] += curr['map@1']
        metrics['map@5'] += curr['map@5']
        metrics['map@10'] += curr['map@10']
        metrics['diversity@5'] += curr['diversity@5']
        metrics['diversity@10'] += curr['diversity@10']
        metrics['epc@5']+= curr['epc@5']
        metrics['epc@10'] += curr['epc@10']
        metrics['epd@5'] += curr['epd@5']
        metrics['unique@1'][curr['unique@1']] = 1
        metrics['unique@5'][curr['unique@5']] = 1
        metrics['unique@10'][curr['unique@10']] = 1
        metrics['nr_users'] += curr['nr_users']
        return metrics
    
    def collect_metrics(self, metrics):
        result = dict()
        result['accuracy'] = np.mean(metrics['accuracy'])
        result['precision@5'] = np.mean(metrics['precision@5'])
        result['recall@5'] = np.mean(metrics['recall@5'])
        result['map@1'] = np.mean(metrics['map@1'])
        result['map@5'] = np.mean(metrics['map@5'])
        result['map@10'] = np.mean(metrics['map@10'])
        result['diversity@5'] = np.mean(metrics['diversity@5'])
        result['diversity@10'] = np.mean(metrics['diversity@10'])
        result['epc@5']=  np.mean(metrics['epc@5'])
        result['epc@10'] = np.mean(metrics['epc@10'])
        result['epd@5'] = np.mean(metrics['epd@5'])
        result['coverage@1'] = np.sum(metrics['unique@1']) / float(self.nr_items)
        result['coverage@5'] = np.sum(metrics['unique@5']) / float(self.nr_items)
        result['coverage@10'] = np.sum(metrics['unique@10']) / float(self.nr_items)
        return result
    
    def evaluate_single_thread(self, model, mode = 'test', max_nr_batches = None):
        metrics = self.init_metrics()
        nr_batches = 0
        batch_size = 128
        dataset = load_dataset(self.dataset, mode).batch(batch_size)
        if max_nr_batches is not None:
            dataset = dataset.take(max_nr_batches)
            
        for batch in dataset:
            x = batch['x']
            user_ids = batch['user_id']
            predictions = model.predict(x, user_ids)
            print('\rBatch nr {} predicted'.format(nr_batches + 1), end='\r')

            result = self.calc_recommendation_metrics(predictions, batch)
            nr_batches += 1
            
            metrics = Evaluation.update_metrics(metrics, result)

        metrics = self.collect_metrics(metrics)
            
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

