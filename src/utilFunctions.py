import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    pe = np.ma.corrcoef(userMovie)
    
    return pe