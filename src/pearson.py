import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):
    pear = sim(userMovie[0], userMovie[1])
    pass

def sim(x, y):

    xSD = np.std(x)
    ySD = np.std(y)
    cov = np.cov(x, y)

    temp = np.corrcoef(x)

    print("correcoef => " , temp , "\n")
    print("p XY => " , cov/xSD*ySD)

    return cov