import talos as ta
import wrangle
from talos.utils import SequenceGenerator
from talos.utils.gpu_utils import parallel_gpu_jobs

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout

x, y = ta.templates.datasets.iris()
x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .5)

def mnist_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=params['activation'], input_shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=['acc', ta.utils.metrics.f1score])

    out = model.fit_generator(SequenceGenerator(x_train,
                                                y_train,
                                                batch_size=params['batch_size']),
                                                epochs=params['epochs'],
                                                validation_data=[x_val, y_val],
                                                callbacks=[],
                                                workers=4,
                                                verbose=0)

    return out, model

p = {'activation':['relu', 'elu'],
     'optimizer': ['AdaDelta'],
     'losses': ['logcosh'],
     'shapes': ['brick'],
     'first_neuron': [32],
     'dropout': [.2, .3],
     'batch_size': [64, 128, 256],
     'epochs': [1]}

parallel_gpu_jobs(0.5)

scan_object = ta.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=mnist_model,
                      experiment_name='test1')