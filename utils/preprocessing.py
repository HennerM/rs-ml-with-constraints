#%%
import tensorflow as tf
import pandas as pd
import numpy as np



def load_data(ratings_csv, movies_csv):

    ratings = pd.read_csv(ratings_csv)
    movies = pd.read_csv(movies_csv)
    movies['mId'] = movies.index
    ratings = ratings.join(movies, on='movieId', rsuffix='_movies')
    return ratings


def create_rating_matrix(ratings):
    nr_users = int(max(ratings['userId']))
    nr_movies = int(max(ratings['mId']))
    rating_matrix = np.zeros((nr_users, nr_movies))
    rating_mask = np.zeros((nr_users, nr_movies), dtype=int)
    for index, row in ratings.iterrows():
        if row['mId'] < nr_movies and row['userId'] < nr_movies:
            rating_matrix[int(row['userId']) - 1, int(row['mId']) - 1] = row['rating']
            rating_mask[int(row['userId']) - 1, int(row['mId']) - 1] = 1
    return rating_matrix, rating_mask


def convert_to_implicit(rating_matrix):
    return rating_matrix - 3.0 >= 0


def augment_unobserved(rating_mask, rate = 0.2):
    # augment each users observations with unobserved

    def activate_ratings(single_row):
        unobserved = single_row == 0
        nr_observed = len(single_row.nonzero()[0])
        new_observations = round(nr_observed * rate)

        single_row[np.random.choice(unobserved.nonzero()[0], new_observations)] = 1
        return single_row

    return np.apply_along_axis(activate_ratings, 1, rating_mask)


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(x, x_mask, y, y_mask):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.



    feature = {
        'x': _int64_feature(x.nonzero()[0]),
        'x_mask': _int64_feature(x_mask.nonzero()[0]),
        'y':  _int64_feature(y.nonzero()[0]),
        'y_mask': _int64_feature(y_mask.nonzero()[0])
    }
    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


ratings = load_data('../../Data/MovieLens/ml-latest-small/ratings.csv', '../../Data/MovieLens/ml-latest-small/ratings.csv')

y, y_mask = create_rating_matrix(ratings)
y = convert_to_implicit(y)
y_mask = augment_unobserved(y_mask)


filename = 'train.tfrecord'
writer = tf.io.TFRecordWriter(filename)
for i in range(rating_matrix.shape[0]):
    writer.write(serialize_example(binary_ratings[i], rating_mask[i]))


