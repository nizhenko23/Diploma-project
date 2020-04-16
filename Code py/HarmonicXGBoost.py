import math
import numpy as np
import matplotlib.pyplot as plot
import random
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, column_or_1d
import LoadAndCheckModel
import xgboost
from MemGridCVSearch import MemGridSearch

in_data = np.arange(0, 4 * math.pi, 4 * math.pi / 400, dtype=np.float32).reshape(-1, 1)
out_data = 2 * np.cos(in_data) + np.cos(4 * np.pi * in_data + np.pi / 3)


plot.ion()
featureNumber = in_data.shape[1]
output_channel = out_data.shape[1]
n_folds = 5
params = {'min_child_weigh': [1,2,3,5,6,7,8 ],
          'gamma': np.arange(0.0001,0.01,0.0001),
          'subsample': np.arange(0.3,0.6,0.05),
          'colsample_bytree': [0.7, 0.75,0.8,0.85,0.9],
          'max_depth': [i for i in range(1,40)],
          'learning_rate':[0.001,0.01,0.05,0.1,0.2,0.5,1]}

model = xgboost.XGBRegressor(eval_metric='rmse')

kf = KFold(n_folds, shuffle=True, random_state=42)#.get_n_splits(in_data)

mem_grid_search = MemGridSearch(model, params, cv=kf, n_jobs=6, verbose=1,refit=True)

mem_grid_search.read_memory('result.xlsx')

for i in range(5000):
    mem_grid_search.fit(in_data, out_data, 50)
    mem_grid_search.save_memory('result.xlsx')
    LoadAndCheckModel.load_and_check(model)