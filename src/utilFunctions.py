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


def normalized_row(user: pd.Series):
    user_mean = user.mean
    # user_normal = user.subtract(user_mean)
    # return user_normal
    pass


def score(userMovies: pd.DataFrame,
          correlation: pd.DataFrame, user_ratings: pd.Series, neighbors: pd.Series, movie_ID):

    # normalized_data(userMovies)
    # User 42 : [Nan ,Nan , 0.45 , Nan , 0.85]

    # if movie_ID in user_ratings:
    #     return user_ratings[movie_ID]

    #Get the average ratings of the user
    ratings_avg = user_ratings.mean
    #Similarity of the ratings of the neighbors to the user that we calculated
    sim_sum = 0
    for n in neighbors:
        sim_sum += correlation.loc[user_ratings[1], n]
    #Similarity times the normalized average ratings of the users
    sim_times_rating = 0
    for n in neighbors:
        sim_times_rating += (correlation.loc[user_ratings[1], n]
                             * (userMovies.loc[movie_ID, n] - userMovies[:, n].mean))
    #Calculate the final score
    score = ratings_avg + (sim_times_rating / sim_sum)

    return score


def scoreItem(df: pd.DataFrame, user: pd.Series
              , neighbor_rating, neighbor_similarity, ratings):
    # We will be using the weighted average method to compute the score.
    # We want to only score items that have not been scored yet.
    # user_norm = normalized_row(user)
    # avg_ratings = user.mean
    #
    # selectTopNeighbors(df)

    # Two dataframes: Neighbors and Ratings

    df_normal = normalized_data(df)

    # df_normal.apply(scoreRow(row))

    pass
