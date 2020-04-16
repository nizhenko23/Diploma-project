import time
import numpy as np
import random
from os import makedirs, path
from shutil import copyfile
import sys
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, skew
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import metrics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
# from plot_learning_curve import plot_learning_curve
import time
import xlsxwriter
# import POD
import tensorflow as tf

y_data = pd.read_csv('pwdb_haemod_params.csv')
y_data.drop('Velocity[k] (m/s)', axis=1, inplace=True)

mean_tensor = tf.reduce_mean(y_data, axis=1, keep_dims=True)
mean_centered_data = tf.subtract(y_data, mean_tensor)

svd_out = tf.svd(mean_centered_data, compute_uv=True, full_matrices=False, name="svd")
s, u, v = (tf.Session()).run(svd_out)

e = (tf.Session()).run(tf.reduce_sum(s))
s_energy = (tf.Session()).run(tf.div(s,e)*100)

coeffs = tf.matmul(tf.transpose(u), y_data)

def create_model(N1=20, N2=20, N3=20, N4=20, P1=1, P2=1, P3=1, lr=0.01):

    model = Sequential([
        Dense(N1, activation='relu', input_dim=220),
        Dropout(P1),
        Dense(N2, activation='relu'),
        Dropout(P2),
        Dense(N3, activation='relu'),
        Dropout(P3),
        Dense(N4, activation='relu'),
        Dense(1, activation='linear')
    ])

    adam = optimizers.Adam(lr)

    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=[metrics.mse])
    return model

model = KerasRegressor(N1=20, N2=20, N3=20, N4=20, P1=0, P2=0, P3=0, lr=0.001, build_fn=create_model,
                       epochs=5000, batch_size=1458, verbose=1)  ###
# print(train_np.shape, y_train.shape)

paramd_dist = {
    "N1": [16, 32, 64, 128, 256],
    "N2": [16, 32, 64, 128, 256],
    "N3": [16, 32, 64, 128, 256],
    "N4": [16, 32, 64, 128, 256],
    "P1": [0.5, 0.4, 0.3, 0.2, 0.1, 0],
    "P2": [0.5, 0.4, 0.3, 0.2, 0.1, 0],
    "P3": [0.5, 0.4, 0.3, 0.2, 0.1, 0],
    # "weight_decay": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-4, 5e-5]
}

start = time.time()
n_iter_search = 20
n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

# score = rmsle_cv(KerasRegressor(**paramd_dist_best))
# print(score.mean(),score.std())

random_search = RandomizedSearchCV(model, param_distributions=paramd_dist, scoring="neg_mean_squared_error",
                                   n_iter=n_iter_search, verbose=2, cv=kf,
                                   n_jobs=5, random_state=5, return_train_score=True)

random_search.fit(x_train, y_data)

# plot_learning_curve(random_search, 'Cross Validation', train_np, y_train, n_jobs=3)
# score = rmsle_cv(random_search)
# print("Score: {:.6f} ({:.6f})\n".format(score.mean(), score.std()))

print(random_search.best_index_)
print(random_search.best_params_)
print(np.sqrt(-random_search.best_score_))
print(random_search.cv_results_)
print(random_search.best_estimator_)

# print("Score: {:.6f} ({:.6f})\n".format(score.mean(), score.std()))
end = time.time()
print('Time of {} iterations: '.format(n_iter_search), end-start)

cv_results = pd.DataFrame.from_dict(random_search.cv_results_)

def xlsxDTwrite(filename, data):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Sheet1')
    writer.save()

xlsxDTwrite('cv_results.xlsx', cv_results)
