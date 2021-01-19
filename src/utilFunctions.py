import numpy as np
import pandas as pd
import os.path
from random import randint


def pearson(moviesUser):
    return moviesUser.corr(method="pearson")


def threshold(t: float, neighbors: int, df: pd.DataFrame):
    #Lower limit for neighbors is 10
    neighbors = max(10, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, row[(row > t)]), axis=1)

    return ret


def selectTop(neighbors: int, df: pd.DataFrame):
    #Lower limit for neighbors is 10
    neighbors = max(10, neighbors)
    
    ret = df.apply(lambda row: seriesLargest(neighbors, row), axis=1)

    return ret


def seriesLargest(neighbors: int, row: pd.Series):
    #Filter out all elements that are below or equal to the threshold
    s = row.nlargest(neighbors + 1, keep='all').head(neighbors + 1).index.to_numpy()

    #Ignore your own value (user3-user3 bijv), which is also why we take neighbors+1 largest values
    #Use row.name to get the index
    s = np.delete(s, np.argwhere(s == row.name))

    #If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values=(0, 0))

    return s


def normalized_data(df: pd.DataFrame):
    df_mean = df.mean(axis=1)
    df_normal = df.subtract(df_mean, axis='rows')
    return df_normal


def score(uM, nn, moviesUser: pd.DataFrame, normalized_matrix: pd.DataFrame, correlation: pd.DataFrame,
          overall_movie_mean: int):
    #Convert to numpy and set properly
    uM1 = uM.to_numpy()
    user_id = uM1[0]
    movie_id = uM1[1]

    #Check if it's already rated.
    if pd.notna(moviesUser[movie_id][user_id]):
        return moviesUser[movie_id][user_id]

    #Average of the user and movie ratings before normalization
    user_ratings_average_unnormalized = moviesUser.loc[user_id].mean(axis=0)
    movie_ratings_average_unnormalized = moviesUser[movie_id].mean(axis=0)
    #We use the normalized dataset here.
    # moviesUser = normalized_matrix
    print(moviesUser)

    if (np.isnan(movie_ratings_average_unnormalized)): 
        movie_ratings_average_unnormalized = overall_movie_mean

    #Calculate the baseline estimate that gets added. 
    #Not 100% sure about the formula
    baseline_estimate = overall_movie_mean + (
            user_ratings_average_unnormalized - overall_movie_mean) + (
            movie_ratings_average_unnormalized - overall_movie_mean)

    #Neighbors of the user.
    neighbors = nn[user_id]

    #Ignore zero-values in NN array, zeros means that there are no neighbors
    neighbors = neighbors[(neighbors > 0)]
    if(neighbors.size < 1):
        return baseline_estimate

    #Similarity of the ratings of the neighbors to the user that we calculated. It's the denominator. What if nan?
    # sim_sum = np.nansum(correlation[user_id])
    sim_sum= 0
    for n in neighbors:
        if pd.notna(moviesUser[movie_id][n]):
            cor = correlation[n][user_id]
            sim_sum += cor
    #Similarity times the normalized average ratings of the users. This is the nominator.
    sim_times_rating = 0
    for n in neighbors:
        #If the neighbors have rated that movie, calculate this.
        if pd.notna(moviesUser[movie_id][n]):
            simxy = correlation[n][user_id]
            avg_movie_rating = moviesUser[movie_id].mean(axis=0)
            ryi = moviesUser[movie_id][n] - avg_movie_rating
            sim_times_rating += (simxy * ryi)

    predicted_score = 0
    #Calculate the final score #rating average
    if sim_sum != 0:
        predicted_score = (sim_times_rating / sim_sum)

    #To get what the user would rate a movie out of 5
    predicted_rate = predicted_score + baseline_estimate

    #We can put limits too, e.g. cut off at above 5 and below 1.
    predicted_rate = max(min(round(predicted_rate, 2), 5), 1)

    return predicted_rate


def rating(predictions: pd.DataFrame, utilMatrix: pd.DataFrame, nn, moviesUser: pd.DataFrame):
    #Some usefull variables
    normal_um = normalized_data(moviesUser)
    overall_movie_mean = moviesUser.mean().mean()
    # print(moviesUser)
    # average_user_mean = moviesUser.mean(axis=0)
    # print(average_user_mean)
    # average_movie_mean = moviesUser.mean(axis=1)
    # print(average_movie_mean)
    newPredictions = predictions.apply(lambda uM: 
                score(uM, nn, moviesUser, normal_um, utilMatrix, overall_movie_mean), axis=1)

    return newPredictions


