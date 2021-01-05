import numpy as np
import pandas as pd
import os.path
from random import randint

def pearson(userMovie):

    pe = np.ma.corrcoef(np.ma.masked_invalid(userMovie))
    print(pe[:40, :50])
    return pe