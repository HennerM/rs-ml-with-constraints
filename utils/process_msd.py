import pandas as pd
import numpy as np
import os
from common import save_df_to_records, split_train_test_validate_df, augment_negative_sampling

def convert_to_implicit(play_count):
    # return play_count > 1#
    return True


def group_and_transform(ratings):
    ratings = ratings[['user_id', 'item_id', 'play_count']]
    pd_values = ratings.sort_values('user_id').values.T
    user_indices = pd_values[0]
    values = np.dstack((pd_values[1], pd_values[2]))[0]
    ukeys, index = np.unique(user_indices, True)
    arrays = np.split(values, index[1:])
    df = pd.DataFrame({'userId': ukeys, 'rated': arrays})
    df['positive'] = df['rated'].map(lambda x: [y[0] for y in x if convert_to_implicit(y[1])])
    return df

if __name__ == "__main__":
    ratings =  pd.read_csv(os.path.dirname(__file__) + '../../Data/msd/msd_data.csv')
    nr_items = ratings['item_id'].max() + 1
    nr_users = ratings['user_id'].max() + 1
    train_ratings, test_ratings, validate_ratings = split_train_test_validate_df(ratings)
    print("Total ratings:", len(ratings))
    print("Train ratings:", len(train_ratings))
    print("Validation ratings:", len(validate_ratings))
    print("Test ratings:", len(test_ratings))
    print("Items:", nr_items)
    print("User:", nr_users)

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


    save_df_to_records(train_data, os.path.dirname(__file__) + '../../Data/msd/train.tfrecords')
    save_df_to_records(test_data, os.path.dirname(__file__) + '../../Data/msd/test.tfrecords')
