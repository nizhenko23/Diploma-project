from __future__ import absolute_import, division, print_function, unicode_literals
import time
import matplotlib.pyplot as plot
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split, ShuffleSplit, LeaveOneOut
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

tf.config.experimental.list_physical_devices('GPU')

data = pd.read_csv('pwdb_haemod_params.csv', sep=',')
del data['Subject Number']
data = data.apply(pd.to_numeric)
cols = data.columns.tolist()
cols = cols[24:] + cols[:24]
data = data[cols]
print(data.describe())

X, y = data.iloc[:, :28], data.iloc[:, 28:]
in_data = X.values
out_data = y.values

cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=123)

def create_model(N1=20, N2=20, N3=20, N4=20, P1=1, P2=1, P3=1, lr=0.01):

    model = Sequential([
        Dense(N1, activation='relu', input_dim=28),
        Dropout(P1),
        Dense(N2, activation='relu'),
        Dropout(P2),
        Dense(N3, activation='relu'),
        Dropout(P3),
        Dense(N4, activation='relu'),
        Dense(4, activation='linear')
    ])

    adam = optimizers.Adam(lr)

    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=[metrics.mse])
    return model


model = KerasRegressor(N1=20, N2=20, N3=20, N4=20, P1=1, P2=1, P3=1, lr=0.01, build_fn=create_model,
                       epochs=20, batch_size=200, verbose=1)

paramd_dist = {
    "N1": [16, 32, 64, 128, 256],
    "N2": [16, 32, 64, 128, 256],
    "N3": [16, 32, 64, 128, 256],
    "N4": [16, 32, 64, 128, 256],
    "P1": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "P2": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "P3": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-3, 5e-4, 5e-5],
}

start = time.time()
n_iter_search = 50
nfolds = 4
kf = KFold(nfolds, shuffle=True, random_state=123)#.get_n_splits(in_data)

random_search = RandomizedSearchCV(model, param_distributions=paramd_dist, scoring="neg_mean_squared_error",
                                   n_iter=n_iter_search, verbose=2, cv=cv,
                                   n_jobs=5, random_state=5, return_train_score=True)

random_search.fit(in_data, out_data)

end = time.time()
print('Time of {} iterations: '.format(n_iter_search), end-start)
# plot.show()

cv_results = pd.DataFrame.from_dict(random_search.cv_results_)

def xlsxDTwrite(filename, data):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Sheet1')
    writer.save()

xlsxDTwrite('cv_results.xlsx', cv_results)
