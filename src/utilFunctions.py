import numpy as np
import pandas as pd
import os.path
from random import randint


def pearson(userMovie):
    return userMovie.corr(method="pearson")


def threshold(t: float, neighbors: int, df: pd.DataFrame):
    # lower limit for neighbors is 50
    neighbors = max(50, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, t, row), axis=1)

    return ret


def selectTopNeighbors(neighbors, df):
    ret = df.apply(lambda row: seriesLargestTop(neighbors, row), axis=1)

    return ret


def seriesLargestTop(neighbors: int, row: pd.Series):
    s = row.nlargest(neighbors + 1, keep='all').head(neighbors + 1).index.to_numpy()
    s = np.delete(s, np.argwhere(s == row.name))
    # If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values=(0, 0))
    return s


def seriesLargest(neighbors: int, t: float, row: pd.Series):
    # Filter out all elements that are below or equal to the threshold
    s = row[(row > t)].nlargest(neighbors + 1, keep='all').head(neighbors + 1).index.to_numpy()

    # print("\n\n____\n", s, " size=", s.size)

    # Ignore your own value (user 3 - user 3 bijv), which is also why we take neighbors+1 largest values
    # Use row.name to get the index
    s = np.delete(s, np.argwhere(s == row.name))

    # print("\n", s , " size=", s.size)

    # If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values=(0, 0))

    # print ("=> \n", s , " size=", s.size)

    return s


def normalized_data(df: pd.DataFrame):
    df_mean = df.mean(axis=1)
    df_normal = df.subtract(df_mean, axis='rows')
    # print(df_normal)
    return df_normal


def score(userMovies: pd.DataFrame,
          correlation: pd.DataFrame, user_ID, user_ratings: pd.Series, neighbors: pd.Series, movie_ID):
    # normalized_data(userMovies)
    # User 42 : [Nan ,Nan , 0.45 , Nan , 0.85]

    # if movie_ID in user_ratings:
    #     return user_ratings[movie_ID]

    # print(user_ratings[movie_ID])

    # Get the average ratings of the user
    ratings_avg = user_ratings.mean(axis=0)
    # print(neighbors)
    # Similarity of the ratings of the neighbors to the user that we calculated
    sim_sum = 0
    for n in neighbors:
        sim_sum += correlation[user_ID][n]
    # Similarity times the normalized average ratings of the users
    sim_times_rating = 0
    for n in neighbors:
        if pd.notna(userMovies[n][movie_ID]):
                        print("Neighbor num :", n, "Movie ", userMovies[n][movie_ID])
                        sim_times_rating += (correlation[user_ID][n] * (userMovies[n][movie_ID] - userMovies[n].mean(axis=0)))
    # Calculate the final score
    predicted_score = ratings_avg + (sim_times_rating / sim_sum)

    return predicted_score
