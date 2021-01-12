import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie : pd.DataFrame):

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