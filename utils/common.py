from typing import Tuple

import os
import tensorflow as tf
from numpy.core.multiarray import ndarray

movie_lens = {
    'feature_description': {
        'x': tf.io.VarLenFeature(tf.int64),
        'mask': tf.io.VarLenFeature(tf.int64),
        'y': tf.io.VarLenFeature(tf.int64),
        'userId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    },
    'train': {
        'records': 610,
        'filenames': [os.path.dirname(__file__) + '/../../Data/MovieLens/ml-latest-small/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filenames': (os.path.dirname(__file__) + '/../../Data/MovieLens/ml-latest-small/test.tfrecords')
    },
    'item_features': os.path.dirname(__file__) + '/../../Data/MovieLens/movie_features.npz',
    'user': 610,
    'dimensions': 10379,
}


def load_dataset(ds: dict, edition = 'train') -> tf.data.Dataset:
    def parse_example(example_proto) -> dict:
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        return {'x': x, 'mask': mask, 'user_id': parsed['userId']}

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)
