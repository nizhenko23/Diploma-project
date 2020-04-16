import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from MemGridCVSearch import MemGridSearch
from sklearn.model_selection import train_test_split

# load data
#
#
#
#

data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

params = {
    "objective":"reg:linear",
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}
#
# cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                    num_boost_round=100, early_stopping_rounds=5,
#                    metrics="rmse", as_pandas=True, seed=123)
#
# cv_results.tail()
#

model = xgb.XGBRegressor()
loo = LeaveOneOut()
#
# params = {"objective": "reg:linear",
#           'colsample_bytree': np.arange(0.05, 1, 0.05),
#           'learning_rate': np.arange(0.0001, 1, 0.05),
#           'max_depth': np.arange(1, 15, 1),
#           'alpha': np.arange(5, 40, 5)
#          }

mem_grid_search = MemGridSearch(estimator=model, param_grid=params, cv=LeaveOneOut(), n_jobs=4, verbose=1, refit=True)
mem_grid_search.read_memory('results.xlsx')

for i in range(5000):
    mem_grid_search.fit(X, y, 12)
    mem_grid_search.save_memory('results.xlsx')
