# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:43:11 2018

@author: lucky
"""

# Recurrent Neural networks

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set.
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# The first colon states that we need all the columns, from the beginning to the end.
# The : Used to omit upper bounds, so use only column 1.
training_set = dataset_train.iloc[:, 1:2].values

# From Rnn and if there is sigmoid function in output layer, it is recommended to use Normalization
# that would be in the MinMax scale class
from sklearn.preprocessing import MinMaxScaler

# All the scaled stock prices lie between 0 and 1 as Max - Min will always be greater than input - Min
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create the dataset wirh 60 timestamps and 1 output.
X_train = []
y_train = []
for i in range(60, 1258):
	X_train.append(training_set_scaled[i-60: i, 0])
	y_train.append(training_set_scaled[i, 0])

# Convert the appended values to the numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

'''
Observation of row 0.

Understanding the values of X_train and y_train.
In the values of X_train, the first values in the obeservation point to the 
stock price on the 60th day. and rest all the colums of 59 have all the 60 values
are the previous 60 stock prices before the stock price on 60th day.

Observation of row 1

It has the values of stock price of the 61 stock prices, and previous 60 values untill 1.

As a lamen example, all the stock prices in the X_train 1st row will be used to predict the 60+1 row in y_train
There are also ways to add indicators which include close, high, low and so on.

These new indicators will be added as a new dimension.
'''

# Reshaping the data, adding new dimension to add new indicators.
'''
To add input shapes the tensor we will alter the tensor to take up a 3D shape so that we can 
add new indicators as dimensions.

This will be listed in the keras documentation under Recurrent layers -> Input Shapes
input shape is what is expected by the RNN

Arguments: Batch_size: ex: total observations we have from 2012 to 2016,
           Time steps: 60 steps, ideal value
		     input_dim: will be where we can add additional indicators, close, or other stock prices  from other companies.
			  example samsung and apple might be corelated, so according to corelation we can add indicators.

You can use X_train.shape[0] it will give the no of rows or lines ie stock prices.
You can use X_train.shape[1] it will give the no of columns or lines ie timesteps.

If there are more no of indicaotrs we have to change you will have to change the value '1'
according to the no of indicators you are using.

This will have the right structure expected by the network, in this case it indicates,
the rows -> Stock prices, columns -> timesteps and an additional indicator.

There is a file AddIndicators.py, take a look at it to see how to add multiple indicators.

You had a good question where in you compare stocks of 2 companies that have a corelation. 
But there is no way that the total no of stock indicators of one comapny are more than the other as they are based off timestamps.
'''
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the RNN

# Importing the various Libraries needed
# We are using Dropout to avoid overfitting, making a stacked LSTM.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN.
# Initialize the RNN as a squence or a computational graph.
# Later we are going to use pytorch to make computational graphs.

# Regressor represents a sequence of layers.
# Becuase unlike ANN and CNN we are expecting a continuous outputs.
'''
Regression is about predicting a continuous value where as
Classification is used to predict a category.
'''
regressor = Sequential()

# Adding an LSTM layer and Dropout Regularization.
'''
----IMPORTANT-----
There are 3 very important arguments in the LSTM class
They are: 
	1. Units: No of LSTM cells or memory cells you want in the LSTM, a relevant no, or nuerons.
	2. Return sequences: True, as we are making a stacked LSTM, it has several layers.
	When you add more layers you will set the return sequences is set to True.
	Once you are not going to add more stacked LSTM layers, set the value to False, set by default.
	3. Input Shape: Same shape as that of the value X_train.
	You will have to send in the values of X_train.shape[1] and no of indicators as 
	the value of X_train.shape[0] will be taken in by default.
'''
# layer 1
# You will have to add units basing on the problem, since stocks are complex
# it is best to have hight dimensionality and nuerons.
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# The ideal no to use would be 20perc of the nuerons.
regressor.add(Dropout(0.2))

# Adding additional LSTM layers
# Layer 2 of LSTM
# No need to specify input shape as it knows there are 50 neurons that act as input.
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Layer 3 of LSTM
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Layer 4 of LSTM
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
# It has only 1 dimesion, no of nuerons in the output layer, total outputs.
regressor.add(Dense(units = 1))

# Compiling the RNN
# rmsprop is recommended, but there is always a choice, check keras documentation.
# adam optimizer is always a safe choice.

# Regression being continuous we cannot use binary cross entropy. We use mean squared error instead
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting teh RNN to the training set.
# Experimental values. You can change em to make it better.
# batch_training is used to perform an update on weights after certain no of inputs.
# Experiment with various values.
# Check for convergence in the losses as the loss value decreases after every back propogation.
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 Making predictions and visualizing the data in a graph to see how accurate the RNN is.

# Getting the real stock price of 2017
# this can be done from the google_csv file
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# There are some mistakes that are to be avoided.
'''
Key points to be understood.
1. The model to be able to predict at time t+1 it needs all 60 precious stock prices leading to it and other following.
2. In order to get each day of Jan the 60 previous stock prices o fthe 60 previous days, we will need both training and test set.
3. Make it a point to never change the test values. Leave them be. Only scael the inputs and not the test values
'''
# Horizontal axis is labelled by 1
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(60, 80):
	X_test.append(inputs[i-60: i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results.
'''
Models cannot react properly to strong non linear changes in stock time irregularity.
'''
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()





















 




