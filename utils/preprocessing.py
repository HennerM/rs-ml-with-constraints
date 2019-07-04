#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import os



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


def hold_back_ratings(y, mask, hold_back_factor=0.2):
    x = np.array(y, copy=True)
    hold_back = list()
    for i in range(x.shape[0]):
        rating_indices = np.where(mask[i] == True)[0]
        nr_ratings = len(rating_indices)
        hold_back.append(np.random.choice(rating_indices, int(nr_ratings * hold_back_factor), replace=False))
        x[i, hold_back[i]] = False
    hold_back = np.asarray(hold_back)
    return x, hold_back


def split_train_test_validate(nr_users, train_split=0.8, validation_split=0.1):
    all_indices = np.arange(nr_users)
    nr_train = int(nr_users * train_split)
    train_indices = np.sort(np.random.choice(all_indices, nr_train, replace=False))
    nr_validation = int(nr_users * validation_split)
    rest = np.setdiff1d(all_indices, train_indices)
    validation_indices = rest[0:nr_validation - 1]
    test_indices = rest[nr_validation:]

    return train_indices, validation_indices, test_indices


def serialize_example(x, y, mask, hold_back = None):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.



    feature = {
        'x': _int64_feature(x.nonzero()[0]),
        'y':  _int64_feature(y.nonzero()[0]),
        'mask': _int64_feature(mask.nonzero()[0]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_to_records(x, mask, y, filename):
    writer = tf.io.TFRecordWriter(filename)
    for i in range(x.shape[0]):
        writer.write(serialize_example(x[i], mask[i], y[i]))


if __name__ == "__main__":
    ratings = load_data('../../Data/MovieLens/ml-latest-small/ratings.csv', '../../Data/MovieLens/ml-latest-small/movies.csv')

    y, mask = create_rating_matrix(ratings)
    y = convert_to_implicit(y)
    mask = augment_unobserved(mask)
    x, held_back = hold_back_ratings(y, mask)
    train_indices, validation_indices, test_indices = split_train_test_validate(x.shape[0])

    train_x = x[train_indices]
    train_y = y[train_indices]
    train_mask = mask[train_indices]

    validate_x = x[validation_indices]
    validate_y = y[validation_indices]
    validate_mask = mask[validation_indices]

    test_x = x[test_indices]
    test_y = y[test_indices]
    test_mask = mask[test_indices]

    save_to_records(train_x, train_mask, train_y, os.path.abspath('../../Data/MovieLens/train.tfrecords'))
    save_to_records(validate_x, validate_mask, validate_y, os.path.abspath('../../Data/MovieLens/validation.tfrecords'))

    np.savez(os.path.abspath('../../Data/MovieLens/test.npz'), x=test_x, y=test_y, mask=test_mask, held_back=held_back)
