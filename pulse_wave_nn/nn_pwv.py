import numpy as np
import pandas as pd
from sklearn import preprocessing
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

# df1 = pd.read_csv('age_hr_sbp_dbp.csv', names=['age', 'hr', 'sbp', 'dbp'])
# df2 = pd.read_csv('pwv_bf.csv', names=['pwv'])
# df = pd.concat([df1, df2], axis=1, join='inner')
# df.to_csv('data.csv', index=False)


# # Инициализация X массива из df
# X = df.iloc[:, 0:-1].values
# # Нормализация колонок массива X
# X_prenorm_min = X.min(0)
# X_prenorm_ptp = X.ptp(0)
# X = (X - X_prenorm_min) / X_prenorm_ptp
# # Инициализация Y массива из df
# Y = df.iloc[:, -1].values

df = pd.read_csv('data.csv') #,usecols = ["Age (years)"," Heart Rate (bpm)"," Systolic BP (mmHg)"," Diastolic BP (mmHg)"," height (m)"," weight (kg)"," Body Mass Index (kn/m2)"," brachial-femoral PWV (m/s)"])
n_inp = len(df.columns) - 1
df_shuffled = df.sample(frac=1).reset_index(drop=True)

normalized = ((df_shuffled - df_shuffled.min())/(df_shuffled.max() - df_shuffled.min()))
normalized2 = preprocessing.normalize(df_shuffled)

df_normalized = pd.DataFrame(data=normalized, columns=['age', 'hr', 'sbp', 'dbp', 'pwv'])
outname ='pwv'
Y = df_normalized[outname].values
df_X = df_normalized.drop(columns=[outname])
X = df_X.values

def pwv_model(x_train, y_train, x_val, y_val, params):
    model = keras.Sequential()

    model.add(keras.layers.Dense(params['first_neuron'], input_dim=n_inp,
                    activation='relu'))
    model.add(keras.layers.Dropout(params['dropout']))

    # Скрытые слои
    for i in range(params['hidden_layers']):
        print(f"adding layer {i + 1}")
        model.add(keras.layers.Dense(params['hidden_neuron'], activation='relu'))
        model.add(keras.layers.Dropout(params['dropout']))

    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation=params['last_activation']))

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
ta_params['lr'] = [0.001, 0.01, 0.0005]
ta_params['first_neuron'] = [4, 8, 16, 32]
ta_params['hidden_neuron'] = [16, 32, 64]
ta_params['hidden_layers'] = [0, 1, 2, 3, 4]
ta_params['batch_size'] = [32, 64, 128]
ta_params['epochs'] = [500]
ta_params['dropout'] = (0, 0.5, 6)
ta_params['optimizer'] = [keras.optimizers.Adam]
ta_params['losses'] = ['mean_squared_error']
ta_params['activation'] = ['relu']
ta_params['last_activation'] = ['softmax', 'linear']

ta_scan = ta.Scan(
    x=X,
    y=Y,
    val_split=0.2,
    seed=32,
    model=pwv_model,
    params=ta_params,
    experiment_name='pwv_experiment_column_normalized',
    clear_session=True
)
