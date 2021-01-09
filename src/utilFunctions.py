import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    return userMovie.corr(method="pearson")

def threshold(t : float, neighbors :int, df : pd.core.frame.DataFrame):

    #lower limit for neighbors is 50
    neighbors = max(50, neighbors)

    ret = df.apply(lambda row: seriesLargest(neighbors, t, row), axis=1)

    return ret

def selectTop(n, df):

    #

    pass

def seriesLargest(neighbors : int, t : float, row : pd.Series):

    s = row[(row > t)].nlargest(neighbors, keep='all').head(neighbors).index.to_numpy()
    s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values = (0,0))

    return s