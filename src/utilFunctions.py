import numpy as np
import pandas as pd
import os.path
from random import randint


def pearson(userMovie):
    return userMovie.corr(method="pearson")


def threshold(t : float, neighbors :int, df : pd.DataFrame):

    #Lower limit for neighbors is 20
    neighbors = max(20, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, row[(row > t)]), axis=1)

    return ret


def selectTop(neighbors, df):

    ret = df.apply(lambda row: seriesLargest(neighbors, row), axis=1)

    return ret


def seriesLargest(neighbors : int, row : pd.Series):

    #Filter out all elements that are below or equal to the threshold
    s = row.nlargest(neighbors+1, keep='all').head(neighbors+1).to_numpy()

    #Ignore your own value (user3-user3 bijv), which is also why we take neighbors+1 largest values
    #Use row.name to get the index
    s = np.delete(s, np.argwhere(s==row.name))

    #If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values = (0,0))

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
          correlation: pd.DataFrame, ratings: pd.Series, neighbors: pd.Series, movie):
    # normalized_data(userMovies)
    # User 42 : [Nan ,Nan , 0.45 , Nan , 0.85]

    if movie in ratings:
        return ratings[movie]

    ratings_avg = ratings.mean
    sim_sum = 0
    for n in neighbors:
        sim_sum += correlation.at[movie, n]

    sim_times_rating = 0
    for n in neighbors:
        sim_times_rating += (correlation.at[movie, n]
                             * (userMovies.at[movie, n] - userMovies[n].mean))

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
