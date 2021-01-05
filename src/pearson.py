import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):
    return sim(userMovie.iloc[0], userMovie.iloc[1])

def sim(x, y):

    xSD = np.std(x)
    ySD = np.std(y)
    cov = np.cov(x, y)

    temp = np.corrcoef(x, y)

    print("correcoef => " + temp + "\n")
    print("p XY => " + cov/xSD*ySD)

    return cov