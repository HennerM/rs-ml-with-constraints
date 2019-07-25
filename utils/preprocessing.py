#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def load_data(ratings_csv, movies_csv):

    ratings = pd.read_csv(ratings_csv)
    movies = pd.read_csv(movies_csv)
    nr_items = movies.count()[0]
    ratings = ratings.merge(movies, on='movieId',how='inner')
    return ratings, nr_items


def create_rating_matrix(ratings, dimensions):
    nr_users = int(ratings['userId'].max())
    print("Nr users:", nr_users)
    print("Nr dimensions:", dimensions)
    rating_matrix = np.zeros((nr_users, dimensions), dtype=np.bool)
    rating_mask = np.zeros((nr_users, dimensions), dtype=np.bool)
    for row in ratings.itertuples():
        if row.mId < dimensions and row.userId < nr_users:
            rating_matrix[int(row.userId) - 1, int(row.mId)] = convert_to_implicit(row.rating)
            rating_mask[int(row.userId) - 1, int(row.mId)] = True
    return rating_matrix, rating_mask


def convert_to_implicit(rating_matrix):
    return rating_matrix >= 3.0


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


def augment_negative_sampling(rated, max_item, negative_sampling_rate = 1.5):
    nr_rated = len(rated)
    sample = np.random.randint(0, max_item, size=int(nr_rated * negative_sampling_rate)).tolist()
    return sample + rated

def group_and_transform(ratings):
    ratings = ratings[['userId', 'mId', 'rating']]
    pd_values = ratings.sort_values('userId').values.T
    user_indices = pd_values[0]
    values = np.dstack((pd_values[1], pd_values[2]))[0]
    ukeys, index = np.unique(user_indices, True)
    arrays = np.split(values, index[1:])
    df = pd.DataFrame({'userId': ukeys, 'rated': arrays})
    df['positive'] = df['rated'].map(lambda x: [y[0] for y in x if convert_to_implicit(y[1])])
    return df

def split_train_test_validate_df(ratings, train_split=0.8, test_split=0.15):
    random_nrs = np.random.rand(len(ratings))
    train_indices = random_nrs < train_split
    test_indices = (random_nrs > train_split) & (random_nrs < (train_split + test_split))
    validate_indices = random_nrs >  (train_split + test_split)
    train = ratings.loc[train_indices]
    test = ratings.loc[test_indices]
    validate = ratings.loc[validate_indices]
    return train, test, validate

def serialize_example(x, y, mask, hold_back = None):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.



    feature = {
        'x': _int64_feature(x.nonzero()[0]),
        'mask': _int64_feature(mask.nonzero()[0]),
    }

    if hold_back is not None:
        feature['held_back'] = _int64_feature(hold_back)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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



if __name__ == "__main__":
    ratings, dimensions = load_data(os.path.dirname(__file__) + '../../Data/MovieLens/ml-latest-small/ratings.csv', os.path.dirname(__file__) + '../../Data/MovieLens/ml-20m/movie_mapping.csv')
    ratings['userId'] -= 1
    print("Nr of ratings:", ratings['rating'].count())
    nr_items = ratings['mId'].max() + 1
    nr_users = ratings['userId'].max() + 1
    print("Items:", nr_items)
    print("User:", nr_users)
    train_ratings, test_ratings, validate_ratings = split_train_test_validate_df(ratings)
    print("Train ratings:", len(train_ratings))
    print("Validation ratings:", len(validate_ratings))
    print("Test ratings:", len(test_ratings))

    train_ratings = group_and_transform(train_ratings)
    train_ratings['rated'] = train_ratings['rated'].map(lambda x: augment_negative_sampling([y[0] for y in x], nr_items))

    validate_ratings = group_and_transform(validate_ratings)
    validate_ratings['rated'] = validate_ratings['rated'].map(lambda x: [y[0] for y in x])

    train_data = train_ratings.merge(validate_ratings, how='left', on='userId', suffixes=('', '_test'))

    test_user_sample = train_ratings.loc[np.random.rand(len(train_ratings)) > 0.8]
    test_ratings = group_and_transform(test_ratings)
    test_ratings['rated'] = test_ratings['rated'].map(lambda x: [y[0] for y in x])


    test_data =test_user_sample.merge(test_ratings, how='inner', on='userId', suffixes=('', '_test'))
    test_data = test_data.loc[test_data.positive_test.apply(lambda x: len(x) > 0)]


    save_df_to_records(train_data, os.path.dirname(__file__) + '../../Data/MovieLens/ml-latest-small/train.tfrecords')
    save_df_to_records(test_data, os.path.dirname(__file__) + '../../Data/MovieLens/ml-latest-small/test.tfrecords')
