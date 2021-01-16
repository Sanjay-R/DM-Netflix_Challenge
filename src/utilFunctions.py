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
    return df_normal


def score(user_movies_matrix: pd.DataFrame, normalized_matrix: pd.DataFrame, correlation: pd.DataFrame, user_id,
          movie_id, overall_movie_mean: int):
    # Average of the user and movie ratings before normalization
    user_ratings_average_unnormalized = user_movies_matrix[user_id].mean(axis=0)
    movie_ratings_average_unnormalized = user_movies_matrix.loc[movie_id].mean(axis=0)
    # We use the normalized dataset here.
    user_movies_matrix = normalized_matrix
    active_user_ratings = user_movies_matrix[user_id]

    # We get all the neighbors.
    neighbors_matrix = selectTopNeighbors(50, correlation)
    # Neighbors of the user.
    neighbors = neighbors_matrix[user_id]

    # Check if it's already rated.
    if pd.notna(active_user_ratings[movie_id]):
        return active_user_ratings[movie_id]

    # Get the average ratings of the user
    ratings_avg = active_user_ratings.mean(axis=0)

    # Similarity of the ratings of the neighbors to the user that we calculated. It's the denominator. What if nan?
    sim_sum = 0
    # print(neighbors)
    for n in neighbors:
        if n == 0: break
        sim_sum += correlation[user_id][n]

    # Similarity times the normalized average ratings of the users. This is the nominator.
    sim_times_rating = 0
    for n in neighbors:
        if n == 0: break
        # If the neighbors have rated that movie, calculate this.
        if pd.notna(user_movies_matrix[n][movie_id]):
            sim_times_rating += (correlation[user_id][n] * (
                    user_movies_matrix[n][movie_id] - user_movies_matrix[n].mean(axis=0)))

    # Calculate the final score #rating average
    predicted_score = (sim_times_rating / sim_sum)
    # Calculate the baseline estimate that gets added. Not 100% sure about the formula
    baseline_estimate = overall_movie_mean + (user_ratings_average_unnormalized - overall_movie_mean) + (
            movie_ratings_average_unnormalized - overall_movie_mean)

    # To get what the user would rate a movie out of 5

    predicted_rate = predicted_score + baseline_estimate

    # We can put limits too, e.g. cut off at above 5 and below 1.

    return predicted_rate
