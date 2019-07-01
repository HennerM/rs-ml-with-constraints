import tensorflow as tf
from models.ConstraintAutoRec import ConstraintAutoRec
import datetime



movie_lens = {
    'feature_description': {
    'rating': tf.io.VarLenFeature(tf.int64),
    'mask': tf.io.VarLenFeature(tf.int64)
    },
    'records': 610,
    'dimensions': 9018,
    'filenames': ['train.tfrecord']

}


def load_dataset(ds: dict):
    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, ds['feature_description'])
        ratings = tf.sparse.to_indicator(parsed['rating'], ds['dimensions'])
        return ratings, tf.sparse.to_indicator(parsed['mask'], ds['dimensions'])

    return tf.data.TFRecordDataset(ds['filenames']).map(parse_example)

today = datetime.date.today()
model = ConstraintAutoRec(movie_lens['dimensions'])
model.train(load_dataset(movie_lens), movie_lens['records'])
model.save('saved/' + str(today) + '/')

