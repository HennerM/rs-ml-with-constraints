from typing import Tuple

import os
import tensorflow as tf
from numpy.core.multiarray import ndarray

movie_lens = {
    'feature_description': {
        'x': tf.io.VarLenFeature(tf.int64),
        'mask': tf.io.VarLenFeature(tf.int64),
        'x_test': tf.io.VarLenFeature(tf.int64),
        'mask_test': tf.io.VarLenFeature(tf.int64),
        'userId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    },
    'train': {
        'records': 138493,
        'filenames': [os.path.dirname(__file__) + '/../../Data/MovieLens/ml-20m/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filenames': (os.path.dirname(__file__) + '/../../Data/MovieLens/ml-20m/test.tfrecords')
    },
    'item_features': os.path.dirname(__file__) + '/../../Data/MovieLens/movie_features.npz',
    'user': 138493,
    'dimensions': 10381,
}


def load_dataset(ds: dict, edition = 'train') -> tf.data.Dataset:
    def parse_example(example_proto) -> dict:
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        if edition == 'test':
            x_test = tf.sparse.to_indicator(parsed['x_test'], ds['dimensions'])
            mask_test = tf.sparse.to_indicator(parsed['mask_test'], ds['dimensions'])
            return {'x': x, 'mask': mask, 'user_id': parsed['userId'], 'x_test': x_test, 'mask_test': mask_test}

        else:
            return {'x': x, 'mask': mask, 'user_id': parsed['userId']}

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)
