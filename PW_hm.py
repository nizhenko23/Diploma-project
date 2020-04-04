import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset = pd.read_csv('pwdb_haemod_params.csv', delimiter=',')
# dataset = dataset.drop(columns='Subject Number')
# dataset = dataset.values
a = dataset.describe()

X = dataset.drop(columns=['PWV_a [m/s]', 'PWV_cf [m/s]', 'PWV_br [m/s]', 'PWV_fa [m/s]'], inplace=True, axis=-1)
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

predictions = model.predict_classes(X)

for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
