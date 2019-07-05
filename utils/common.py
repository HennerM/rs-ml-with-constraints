from typing import Tuple

import os
import tensorflow as tf
from numpy.core.multiarray import ndarray

movie_lens = {
    'feature_description': {
        'x': tf.io.VarLenFeature(tf.int64),
        'mask': tf.io.VarLenFeature(tf.int64),
        'y': tf.io.VarLenFeature(tf.int64),
        'held_back': tf.io.VarLenFeature(tf.int64),
    },
    'train': {
        'records': 610,
        'filenames': [os.path.dirname(__file__) + '/../../Data/MovieLens/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filename': (os.path.dirname(__file__) + '/../../Data/MovieLens/test.tfrecords')
    },
    'item_features': os.path.dirname(__file__) + '/../../Data/MovieLens/movie_features.npz',
    'dimensions': 10381,
}


def load_dataset(ds: dict, edition = 'train') -> tf.data.Dataset:
    def parse_example(example_proto) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        y = tf.sparse.to_indicator(parsed['y'], ds['dimensions'])
        return (x, mask), y

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)

def load_testset(ds: dict) -> tf.data.Dataset:
    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        y = tf.sparse.to_indicator(parsed['y'], ds['dimensions'])
        held_back = tf.sparse.to_indicator(parsed['held_back'], ds['dimensions'])
        return {'x': x, 'mask': mask, 'y': y, 'held_back': held_back}


    return tf.data.TFRecordDataset(ds['test']['filename']).map(parse_example)
