import numpy as np
import pandas as pd
import os.path
from random import randint


def pearson(userMovie):
    return userMovie.corr(method="pearson")


def threshold(t: float, neighbors: int, df: pd.DataFrame):
    # Lower limit for neighbors is 10
    neighbors = max(10, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, row[(row > t)]), axis=1)

    return ret


def selectTop(neighbors: int, df: pd.DataFrame):
    ret = df.apply(lambda row: seriesLargest(neighbors, row), axis=1)

    return ret


def seriesLargest(neighbors: int, row: pd.Series):
    # Filter out all elements that are below or equal to the threshold
    s = row.nlargest(neighbors + 1, keep='all').head(neighbors + 1).index.to_numpy()

    # Ignore your own value (user3-user3 bijv), which is also why we take neighbors+1 largest values
    # Use row.name to get the index
    s = np.delete(s, np.argwhere(s == row.name))

    # If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values=(0, 0))

    return s


def normalized_data(df: pd.DataFrame):
    df_mean = df.mean(axis=1)
    df_normal = df.subtract(df_mean, axis='rows')
    return df_normal


def score(uM, nn, user_movies_matrix: pd.DataFrame, normalized_matrix: pd.DataFrame, correlation: pd.DataFrame,
          overall_movie_mean: int):
    # Convert to numpy and set properly
    uM1 = uM.to_numpy()
    user_id = uM1[0]
    movie_id = uM1[1]

    # Average of the user and movie ratings before normalization
    user_ratings_average_unnormalized = user_movies_matrix[user_id].mean(axis=0)
    movie_ratings_average_unnormalized = user_movies_matrix.loc[movie_id].mean(axis=0)
    # We use the normalized dataset here.
    user_movies_matrix = normalized_matrix
    active_user_ratings = user_movies_matrix[user_id]

    # We get all the neighbors.
    neighbors_matrix = nn
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
    if sim_sum == 0:
        return 0
    predicted_score = (sim_times_rating / sim_sum)
    # Calculate the baseline estimate that gets added. Not 100% sure about the formula
    baseline_estimate = overall_movie_mean + (user_ratings_average_unnormalized - overall_movie_mean) + (
            movie_ratings_average_unnormalized - overall_movie_mean)

    # To get what the user would rate a movie out of 5

    predicted_rate = predicted_score + baseline_estimate

    # We can put limits too, e.g. cut off at above 5 and below 1.

    return predicted_rate


def rating(predictions: pd.DataFrame, utilMatrix: pd.DataFrame, nn, userMovie: pd.DataFrame):
    # Some usefull variables
    normal_um = normalized_data(userMovie)
    overall_movie_mean = userMovie.mean().mean()

    newPredictions = predictions.apply(lambda uM: 
                ratingScore(uM, nn, userMovie, normal_um, utilMatrix, overall_movie_mean), axis=1)
                # score(uM, nn, userMovie, normal_um, utilMatrix, overall_movie_mean), axis=1)

    return newPredictions


### This was mostly for testing and experimenting
def ratingScore(uM, nn, userMovie: pd.DataFrame, normal_um, utilMatrix: pd.DataFrame, 
            overall_movie_mean: int):
    
    # Convert to numpy and set properly
    uM1 = uM.to_numpy()
    userID = uM1[0]
    movieID = uM1[1]

    # Check if movie has already been rated
    if (pd.notna(userMovie[userID][movieID])):
        return userMovie[userID][movieID]  # userMovie[3110][2]

    # Get nearest neighbors of active userID
    buren = nn[userID]
    # Ignore zero-values in NN array, zeros means that there are no neighbors
    buren = buren[(buren > 0)]
    if (buren.size < 1):
        return np.nan

    # Set their default values to 0
    sim_sum = 0
    sim_times_rating = 0

    for n in buren:

        if(pd.notna(userMovie[n][movieID])):
            simxy = utilMatrix[userID][n]
            ryi = userMovie[n][movieID] - userMovie[n].mean(axis=0)

            sim_sum += simxy
            sim_times_rating += (simxy * ryi)

    # The rxi = numerator/denominator = sim_times_rating/sim_sum
    # Check if denominator != 0
    # if(denominator != 0):
    #     nominator = 0
    # else:
    #     return np.nan

    return 0
