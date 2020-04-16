import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
import keras

import talos as ta
from talos import model as ta_model
from talos.metrics.keras_metrics import precision, recall, f1score, matthews, rmse

df1 = pd.read_csv('age_hr_sbp_dbp.csv', names=['age', 'hr', 'sbp', 'dbp'])
df2 = pd.read_csv('pwv_bf.csv', names=['pwv'])
df = pd.concat([df1, df2], axis=1, join='inner')
# df.to_csv('data.csv', index=False)

# Инициализация X массива из df
X = df.iloc[:, 0:-1].values
# Нормализация колонок массива X
X_prenorm_min = X.min(0)
X_prenorm_ptp = X.ptp(0)
X = (X - X_prenorm_min) / X_prenorm_ptp
# Инициализация Y массива из df
Y = df.iloc[:, -1].values

def pwv_model(x_train, y_train, x_val, y_val, params):
    model = keras.Sequential()

    ta_model.hidden_layers(model, params, 0)

    model.add(keras.layers.Dense(1, activation=params['last_activation']))

    model.compile(
        optimizer=params['optimizer'](
            lr=ta_model.normalizers.lr_normalizer(lr=params['lr'], optimizer=params['optimizer'])),
        loss=params['losses'],
        metrics=[
            'acc',
            precision, recall, f1score, matthews, rmse,
            params['losses'],
        ]
    )

    model_fit = model.fit(
        x=x_train, y=y_train,
        epochs=params['epochs'], batch_size=params['batch_size'],
        validation_data=(x_val, y_val),
        verbose=0,
    )

    return model_fit, model

ta_params = dict()
ta_params['lr'] = [0.001]
ta_params['first_neuron'] = [4, 8, 16]
ta_params['hidden_layers'] = [4]
ta_params['batch_size'] = [16]
ta_params['epochs'] = [5000]
ta_params['dropout'] = [0]
ta_params['shapes'] = ['brick']
ta_params['optimizer'] = [keras.optimizers.Adam]
ta_params['losses'] = ['binary_crossentropy']
ta_params['activation'] = ['relu']
ta_params['last_activation'] = ['softmax', 'linear']

ta_scan = ta.Scan(
    x=X,
    y=Y,
    val_split=0.2,
    seed=32,
    model=pwv_model,
    params=ta_params,
    experiment_name='experiment_',
    clear_session=True
)