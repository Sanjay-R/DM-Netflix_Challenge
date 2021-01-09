import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    #

    pass

def threshold(t : float, neighbors :int, df : pd.core.frame.DataFrame):

    #lower limit for neighbors is 50
    neighbors = max(50, neighbors)

    res = np.zeros((df.shape[0], neighbors)) # (amount of rows x amount of neighbors)

    

    for i, row in df.iterrows():
        s = row[(row > t)].nlargest(neighbors, keep='all').head(neighbors).index.to_numpy()
        s = np.pad(s, (0, max(0, (neighbors - s.size))), 'constant', constant_values = (0,0))
        
        # res[i-1] = np.concatenate((res[i-1], alola), axis = 0)
        
        print("\nrow " , i , " = \n", s)

        # for k, l in enumerate(row):
        #     print("l = ", l) #row[(row >= t)]

    #l = 50 x 50 = 2500 values, VALUES
    #k = each column in the row's INDEX
    



    pass

def selectTop(n, df):

    #

    pass

def seriesLargest(buren : int, s : pd.Series):

    #

    pass