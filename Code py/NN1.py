import numpy
import pandas
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
# import numpy.random.common
# import numpy.random.bounded_integers
# import numpy.random.entropy
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf_config.allow_soft_placement = True

# read the data from file, spaces are important
dataframe = pandas.read_csv('dummy_airwave_data.csv') #,usecols = ["Age (years)"," Heart Rate (bpm)"," Systolic BP (mmHg)"," Diastolic BP (mmHg)"," height (m)"," weight (kg)"," Body Mass Index (kn/m2)"," brachial-femoral PWV (m/s)"])

#amount of input parameters
n_inp = len(dataframe.columns) - 1

#shuffle the data

dataframe2 = dataframe.sample(frac=1).reset_index(drop=True)

# split into input (X) and output (Y) variables (non-normalized)

outname =' brachial-femoral PWV (m/s)'


Y = dataframe2[outname].values
dataframe3 = dataframe2.drop(columns=[outname])
X = dataframe3.values

len = len(Y)
border = round(len*0.8)

#use later to validate
Yv = Y[border:len]
Xv = X[border:len, :]

# normalize the data
dataframe2 = ((dataframe2-dataframe2.min())/(dataframe2.max()-dataframe2.min()))
Yn = dataframe2[outname].values
dataframe3 = dataframe2.drop(columns=[outname])
Xn = dataframe3.values

Yln = Yn[0:border]
Xln = Xn[0:border,:]

Yvn = Yn[border:len]
Xvn = Xn[border:len,:]

# define the keras NN model
model = Sequential()
# define number of inputs and nodes in the first layer
model.add(Dense(15, input_dim=n_inp, kernel_initializer='normal', activation='relu'))
for i in range(0, 3): # add 4 layers
    model.add(Dense(15, activation='relu'))
model.add(Dense(1, kernel_initializer='normal')) # output layer
model.compile(loss='mean_squared_error', optimizer='adam') # loss function and optimizer

# fit the model on the dataset
model.fit(Xln, Yln, epochs=250, batch_size=100)

# evaluate the keras model
Yp = model.predict(Xvn)

# a little tricky since Yp is treated as two-dimensional. have to transpose a couple of times
# unnormalize the prediction
Yp = Yp *(Y.max()-Y.min()) + Y.min()
Yp = Yp.transpose()

#calculate RMSE
accuracy = (mean_squared_error(Yv.T, Yp.T))**(0.5)
Yd = Yp - Yv

#Preparing output for the file

outFrame = DataFrame({'Prediction': Yp[0], 'True': Yv,'Diff': Yd[0]})
df_rmse = pandas.DataFrame({'Prediction': ['RMSE'], 'True': ['RMSE'],'Diff': accuracy})

outFrame=outFrame.append(df_rmse,ignore_index = True)
outFrame.to_csv('NN1_out.csv', index=False)


RMSE = accuracy
print('RMSE_pwv: %.2f' % RMSE)