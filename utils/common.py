import tensorflow as tf

movie_lens = {
    'feature_description': {
        'rating': tf.io.VarLenFeature(tf.int64),
        'mask': tf.io.VarLenFeature(tf.int64)
    },
    'train': {
        'records': 610,
        'filenames': ['train.tfrecord']
    },
    'test': {
        'records': 610,
        'filenames': ['train.tfrecord']
    },

}


def load_dataset(ds: dict, edition = 'train'):
    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        ratings = tf.sparse.to_indicator(parsed['rating'], ds['dimensions'])
        return ratings, tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])

    return tf.data.TFRecordDataset(ds[edition]['filenames']).map(parse_example)
