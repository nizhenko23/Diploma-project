from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing
import wrangle
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

out_data = pd.read_csv('pwv_bf.csv', sep=',', header=None)
in_data = pd.read_csv('age_hr_sbp_dbp.csv', sep=',', header=None)
result = pd.concat([in_data, out_data], axis=1, join='inner')

print(result.isnull())
print(result.describe())