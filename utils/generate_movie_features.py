import pandas as pd
import numpy as np
import os

def load_data(genome_csv, tags_csv, movies_csv, genome_relevance_csv):
    genome_tags = pd.read_csv(genome_csv)
    tags = pd.read_csv(tags_csv)
    movies= pd.read_csv(movies_csv)
    genome_relevance = pd.read_csv(genome_relevance_csv)

    genome_relevance['tagId'] = genome_relevance['tagId'] - 1
    genome_tags['tagId'] = genome_tags['tagId'] - 1

    return tags, genome_tags, movies, genome_relevance

def merge_data(tags, genome_tags, movies):
    return tags.merge(genome_tags, on='tag', how='inner').merge(movies, on='movieId', how='inner')

def count_and_normalize(merged):
    tags_group = merged.groupby(['mId', 'tagId'])['userId'].count()
    max_count = tags_group.to_frame().groupby(['mId']).max()['userId']
    return tags_group / max_count

def create_movie_id_mapping(movies, genome_relevance) -> pd.DataFrame:
    movie_mapping = pd.merge(movies, genome_relevance, on='movieId', how='inner')
    movie_mapping = movie_mapping.groupby('movieId')['tagId'].count().reset_index()
    movie_mapping['mId'] = movie_mapping.index
    return movie_mapping[["movieId","mId"]]


def create_known_frequency(movies, ratings):
    relevant_ratings = ratings.merge(movies, how='right', on='movieId')
    movies_by_ratings = relevant_ratings.pivot_table(values='userId', index='mId', aggfunc='count',fill_value=0)
    nr_users = len(relevant_ratings['userId'].unique())
    movies_by_ratings ['userId'] = movies_by_ratings['userId'] / nr_users
    movies_by_ratings.columns = ['known_frequency']
    return movies_by_ratings

if __name__ == "__main__":

    genome_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/genome-tags.csv')
    tags_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/tags.csv')
    movies_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/movies.csv')
    genome_relevance_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/genome-scores.csv')
    ratings_csv = os.path.abspath(__file__ + '/../../../Data/MovieLens/ml-20m/ratings.csv')
    tags, genome_tags, movies, relevance = load_data(genome_csv, tags_csv, movies_csv, genome_relevance_csv)
    ratings = pd.read_csv(ratings_csv)


    # nr_tags = genome_tags['tagId'].count()
    movie_mapping = create_movie_id_mapping(movies, relevance)
    movie_mapping.to_csv(os.path.dirname(__file__) + '../../Data/MovieLens/ml-20m/movie_mapping.csv', index=False)
    # nr_movies = movie_mapping['mId'].count()

    relevance_movies = relevance.merge(movie_mapping, on='movieId', how='inner')

    known_frequency = create_known_frequency(movie_mapping, ratings)
    known_frequency_data = known_frequency.to_numpy()


    movie_features = pd.pivot_table(relevance_movies, values='relevance', index='mId', columns='tagId').to_numpy()
    print("Features shape:", movie_features.shape)

    np.savez(os.path.abspath(__file__ + '/../../../Data/MovieLens/movie_features.npz'), features = movie_features, known_frequency=known_frequency_data)

