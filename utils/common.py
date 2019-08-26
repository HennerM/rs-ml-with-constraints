from typing import Tuple

import os
import tensorflow as tf

ml_feature = {
    'x': tf.io.VarLenFeature(tf.int64),
    'mask': tf.io.VarLenFeature(tf.int64),
    'x_test': tf.io.VarLenFeature(tf.int64),
    'mask_test': tf.io.VarLenFeature(tf.int64),
    'userId': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

movie_lens = {
    'feature_description': ml_feature,
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

ml_small = {
    'feature_description': ml_feature,
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
    'dimensions': 10381,
}




def load_dataset(ds: dict, edition = 'train') -> tf.data.Dataset:
    def parse_example(example_proto) -> dict:
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        x = tf.sparse.to_indicator(parsed['x'], ds['dimensions'])
        mask = tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])
        x_test = tf.sparse.to_indicator(parsed['x_test'], ds['dimensions'])
        mask_test = tf.sparse.to_indicator(parsed['mask_test'], ds['dimensions'])
        return {'x': x, 'mask': mask, 'user_id': parsed['userId'], 'x_test': x_test, 'mask_test': mask_test}

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '>' + '-' * (length - filledLength - 1)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration >= total:
        print()
