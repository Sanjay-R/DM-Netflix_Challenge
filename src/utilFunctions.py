import numpy as np
import pandas as pd
import os.path
from random import randint


def pearson(userMovie):
    return userMovie.corr(method="pearson")


def threshold(t: float, neighbors: int, df: pd.DataFrame):

    #Lower limit for neighbors is 10
    neighbors = max(10, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, row[(row > t)]), axis=1)

    return ret


def selectTop(neighbors: int, df: pd.DataFrame):

    ret = df.apply(lambda row: seriesLargest(neighbors, row), axis=1)

    return ret


def seriesLargest(neighbors: int, row: pd.Series):

    #Filter out all elements that are below or equal to the threshold
    s = row.nlargest(neighbors+1, keep='all').head(neighbors+1).index.to_numpy()

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


def score(user_movies_matrix: pd.DataFrame, correlation: pd.DataFrame, user_id, movie_id):
    # We use the normalized dataset here.

    active_user_ratings = user_movies_matrix[user_id]

    # We get all the neighbors.
    neighbors_matrix = selectTop(50, correlation)
    # Neighbors of the user.
    neighbors = neighbors_matrix[user_id]

    # Check if it's already rated.
    if pd.notna(active_user_ratings[movie_id]):
        return active_user_ratings[movie_id]

    # Get the average ratings of the user
    ratings_avg = active_user_ratings.mean(axis=0)

    # Similarity of the ratings of the neighbors to the user that we calculated. It's the denominator.
    sim_sum = 0
    for n in neighbors:
        sim_sum += correlation[user_id][n]

    # Similarity times the normalized average ratings of the users. This is the nominator.
    sim_times_rating = 0
    for n in neighbors:
        # If the neighbors have rated that movie, calculate this.
        if pd.notna(user_movies_matrix[n][movie_id]):
            sim_times_rating += (correlation[user_id][n] * (
                    user_movies_matrix[n][movie_id] - user_movies_matrix[n].mean(axis=0)))

    # Calculate the final score
    predicted_score = ratings_avg + (sim_times_rating / sim_sum)

    return predicted_score


def rating(predictions: pd.DataFrame, utilMatrix: pd.DataFrame, nn: pd.Series, userMovie: pd.DataFrame):
    newPredictions = predictions.apply(lambda uM: 
                ratingScore(uM, predictions, utilMatrix, nn, userMovie), axis=1)
    pass

def ratingScore(uM, predictions: pd.DataFrame, utilMatrix: pd.DataFrame, nn: pd.Series, userMovie: pd.DataFrame):
    
    #Convert to numpy and set properly
    uM1 = uM.to_numpy()
    userID = uM1[0]
    movieID = uM1[1]

    #Check if movie has already been rated
    if(pd.notna(userMovie[userID][movieID])):
        return userMovie[userID][movieID] #userMovie[3110][2]
    
    #Ignore zero-values in NN array, zeros means that there are no neighbors
    # buren = nn[(nn > 0)]
    # if(buren.size < 1):
    #     return np.nan
    
    # #The rxi = numerator/denominator
    # #Check if denominator != 0
    # denominator = np.sum(utilMatrix[NN])
    # if(denominator != 0):
    #     nominator = 0
    # else:
    #     return np.nan

    return 0
