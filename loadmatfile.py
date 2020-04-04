from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn import preprocessing

y = pd.read_csv('pwv_bf.csv', sep=',', header=None).values
x = pd.read_csv('age_hr_sbp_dbp.csv', sep=',', header=None).values
normalized_x = preprocessing.normalize(x)
print()