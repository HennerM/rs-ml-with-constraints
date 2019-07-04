from typing import Tuple

import os
import tensorflow as tf
from numpy.core.multiarray import ndarray

movie_lens = {
    'feature_description': {
        'x': tf.io.VarLenFeature(tf.int64),
        'mask': tf.io.VarLenFeature(tf.int64),
        'y': tf.io.VarLenFeature(tf.int64),
    },
    'train': {
        'records': 610,
        'filenames': [os.path.dirname(__file__) + '/../../Data/MovieLens/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filename': (os.path.dirname(__file__) + '/../../Data/MovieLens/test.npz')
    },
    'dimensions': 9018,
}


def load_dataset(ds: dict, edition = 'train') -> tf.data.Dataset:
    def parse_example(example_proto) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        y = tf.sparse.to_indicator(parsed['y'], ds['dimensions'])
        return (x, mask), y
    print(ds[edition]['filenames'])

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)
