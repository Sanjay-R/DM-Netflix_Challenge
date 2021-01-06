import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    pe = np.ma.corrcoef(np.ma.masked_invalid(userMovie))
    print(pe[:40, :50])

    print("user 1: " , np.std(userMovie[1]))
    print("user 6038: " , np.std(userMovie[6038]))

    temp = sim(5, 8)
    
    return temp

def sim(x, y):

    return x+y