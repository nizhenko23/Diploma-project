import keras
import talos
import tensorflow
import wrangle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras import losses
from keras.layers import Dense, Dropout
from keras.models import Sequential
from talos.utils.gpu_utils import parallel_gpu_jobs
from talos.model.normalizers import lr_normalizer
from talos.utils.gpu_utils import force_cpu

out_data = pd.read_csv('pwv_bf.csv', sep=',', header=None)
in_data = pd.read_csv('age_hr_sbp_dbp.csv', sep=',', header=None)
result = preprocessing.normalize(pd.concat([in_data, out_data], axis=1, join='inner').values)
x = result[:, :4]
y = result[:, 4]

x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .2)

def pwv_model(x_train, y_train, x_val, y_val, params):
    print(params)

    model = Sequential()

    # Инициализация слоев
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation='relu'))
    model.add(Dropout(params['dropout']))

    # Скрытые слои
    for i in range(params['hidden_layers']):
        print(f"adding layer {i + 1}")
        model.add(Dense(params['hidden_neuron'], activation='relu'))
        model.add(Dropout(params['dropout']))

    # Последний слой
    model.add(Dense(1, activation=params['last_activation']))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                  metrics=['acc', talos.utils.metrics.f1score])

    out = model.fit(x=x_train,
                    y=y_train,
                    validation_data=[x_val, y_val],
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0)
    return out, model

p = {'lr': [0.1, 0.01, 0.001],
     'first_neuron': [16, 32, 64],
     'hidden_neuron': [16, 32, 64, 128, 256],
     'hidden_layers': [1, 2, 3, 4],
     'dropout': [0, .25],
     'batch_size': [32, 64],
     'epochs': [500],
     'last_activation': ['sigmoid']}

parallel_gpu_jobs(0.75)
# Force CPU use on a GPU system
# force_cpu()
scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=pwv_model,
                         experiment_name="pwv")
