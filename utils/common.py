import os
import tensorflow as tf
import numpy as np

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
        'filenames': ['/home/ec2-user/SageMaker/data/MovieLens/ml20m/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filenames': '/home/ec2-user/SageMaker/data/MovieLens/ml20m/test.tfrecords'
    },
    'item_features': '/home/ec2-user/SageMaker/data/MovieLens/movie_features.npz',
    'user': 138493,
    'dimensions': 10381,
}

ml_small = {
    'feature_description': ml_feature,
    'train': {
        'records': 610,
        'filenames': ['/home/ec2-user/SageMaker/data/MovieLens/train.tfrecords']
    },
    'test': {
        'records': 610,
        'filenames': '/home/ec2-user/SageMaker/data/MovieLens/test.tfrecords'
    },
    'item_features': '/home/ec2-user/SageMaker/data/MovieLens/movie_features.npz',
    'user': 610,
    'dimensions': 10381,
}

msd = {
    'feature_description': ml_feature,
    'train': {
        'records': 117966,
        'filenames': ['/home/ec2-user/SageMaker/data/MSD/train.tfrecords']
    },
    'test': {
        'records': 117966,
        'filenames': ['/home/ec2-user/SageMaker/data/MSD/test.tfrecords']
    },
    'item_features': '/home/ec2-user/SageMaker/data/MSD/song_features.npz',
    'user': 117966,
    'dimensions': 6712,
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

def augment_negative_sampling(rated, max_item, negative_sampling_rate = 1.5):
    nr_rated = len(rated)
    sample = np.random.randint(0, max_item, size=int(nr_rated * negative_sampling_rate)).tolist()
    return sample + rated

def split_train_test_validate_df(ratings, train_split=0.8, test_split=0.15):
    random_nrs = np.random.rand(len(ratings))
    train_indices = random_nrs < train_split
    test_indices = (random_nrs > train_split) & (random_nrs < (train_split + test_split))
    validate_indices = random_nrs >  (train_split + test_split)
    train = ratings.loc[train_indices]
    test = ratings.loc[test_indices]
    validate = ratings.loc[validate_indices]
    return train, test, validate


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_df_example(x, mask, userId, x_test, mask_test):
    feature = {
        'x': _int64_feature(map(int, x)),
        'mask': _int64_feature(map(int, mask)),
        'userId': _int64_feature([int(userId)]),
        'x_test': _int64_feature(map(int, x_test) if isinstance(x_test, list) else []),
        'mask_test': _int64_feature(map(int, mask_test) if isinstance(mask_test, list) else []),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def save_df_to_records(df, filename):
    writer = tf.io.TFRecordWriter(filename)
    for df_tuple in df.itertuples():
        writer.write(serialize_df_example(df_tuple.positive, df_tuple.rated, df_tuple.userId, df_tuple.positive_test, df_tuple.rated_test))

