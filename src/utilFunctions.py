import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    #

    pass

def threshold(t, df):

    # for i, j in df.iterrows():
    #     print("\ni => " , i , "\n")
    #     print("\nj => " , j[1])

    
    
    # organ = df[df > t]
    # print(organ)



    pass

def selectTop(n, df):

    #pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)

    # df = df.apply(lambda x: x.sort_values().values)

    df.apply(pd.Series.nlargest, axis=1, n=2)

    print(df)
    pass