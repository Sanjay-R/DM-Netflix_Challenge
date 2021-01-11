import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    return userMovie.corr(method="pearson")

def threshold(t : float, neighbors :int, df : pd.DataFrame):

    #lower limit for neighbors is 50
    neighbors = max(50, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, t, row), axis=1)

    return ret

def selectTop(neighbors, df):

    ret = df.apply(lambda row: seriesLargestTop(neighbors, row), axis=1)

    return ret

def seriesLargestTop(neighbors: int, row : pd.Series):
    s = row.nlargest(neighbors + 1, keep='all').head(neighbors + 1).index.to_numpy()
    s = np.delete(s, np.argwhere(s == row.name))
    # If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values=(0, 0))
    return s

def seriesLargest(neighbors : int, t : float, row : pd.Series):

    #Filter out all elements that are below or equal to the threshold
    s = row[(row > t)].nlargest(neighbors+1, keep='all').head(neighbors+1).index.to_numpy()

    # print("\n\n____\n", s, " size=", s.size)

    #Ignore your own value (user 3 - user 3 bijv), which is also why we take neighbors+1 largest values
    #Use row.name to get the index
    s = np.delete(s, np.argwhere(s==row.name))

    # print("\n", s , " size=", s.size)

    #If there aren't enough neighbors, pad the array with zeros up to neighbors
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values = (0,0))

    # print ("=> \n", s , " size=", s.size)

    return s

def normalized_row(df: pd.DataFrame):
    df_mean = df.mean(axis=1)
    df_normal = df.subtract(df_mean, axis = 'rows')
    # print(df_normal)
    return df_normal

def scoreItem(df,neighbor_rating, neighbor_similarity, ratings):

    #We will be using the weighted average method to compute the score.
    #We want to only score items that have not been scored yet.
    normalized_row(df)


    pass
