
import pandas as pd
import numpy as np
import os

def load_data(genome_csv, tags_csv, movies_csv):
    genome_tags = pd.read_csv(genome_csv)
    tags = pd.read_csv(tags_csv)
    movies= pd.read_csv(movies_csv)
    movies['mId'] = movies.index

    genome_tags['tagId'] = genome_tags['tagId'] - 1

    return tags, genome_tags, movies

def merge_data(tags, genome_tags, movies):
    return tags.merge(genome_tags, on='tag', how='inner').merge(movies, on='movieId', how='inner')

def count_and_normalize(merged):
    tags_group = merged.groupby(['mId', 'tagId'])['userId'].count()
    max_count = tags_group.to_frame().groupby(['mId']).max()['userId']
    return tags_group / max_count

if __name__ == "__main__":

    genome_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/genome-tags.csv')
    tags_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/tags.csv')
    movies_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-latest-small/movies.csv')
    tags, genome_tags, movies = load_data(genome_csv, tags_csv, movies_csv)

    feature_series = count_and_normalize(merge_data(tags, genome_tags, movies))
    nr_tags = genome_tags['tagId'].max() + 1
    nr_movies = movies['movieId'].count()
    print(nr_movies, nr_tags)

    movie_features = np.zeros([nr_movies, nr_tags])
    for index, value in feature_series.items():
        # print(index, value)
        movie_features[index] = value

    np.savez(os.path.abspath(__file__ + '/../../../Data/MovieLens/movie_features.npz'), movie_features)

