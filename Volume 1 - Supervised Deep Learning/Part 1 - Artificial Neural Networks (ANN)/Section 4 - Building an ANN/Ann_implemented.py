# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:39:15 2017

@author: lucky
"""

# currenty working on a classification problem
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#Here the upper bound is excluded.
X = dataset.iloc[:, 3 : 13].values
#y is the output to the dependent variable
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#encoder for changing strings to numbers.
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#this variable dataset is created to avoid the dummy variable trap
#where in the intercept can be used to determine the pattern
#So we select all rokws with colums starting from 1 to ending
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling should be implemented without fail
# As it will ease the computations
# Feature Scaling used to prevent one independent variable dominating another.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(type(X_test[1]))
print(X_test[0])

# Fitting classifier to the Training set

# Importing Keras library and modules for building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
# The object with a rule of a cassifier is the nueral network itself
classifier = Sequential()

# Adding input layer and Hidden layer using Dense function
# Rectifier function for hidden layer and Sigmod to output layer
# Adds, hidden layer which adds inputs in the hidden layer thus defining the no of inputs previously
# output_dim is  the no of nodes in the hidden layer
# Below specified are no of nodes in HL, initialization of weights and activation function i.e rectifier function.
# input dim is a compulsary argument input_dim: no of nodes in input layer, no of indepedent variables
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# The second hidden layer 6 nodes cause (input cols + output cols)/2
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Creating the output layer, uniform used to initialize the weights
# if the output layer has more then 1, you change the output_dim
# Change the activation function to softmax, it is a sigmoid function, but applied to a dependant variable with more than 2 categories
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
 
# Compiling the ANN classifier here
# Adam stands for A method for Stochastic Optimization.
# The loss function is a SUM(diff of real value and predicted value)^2
# A simple neural network with single nueron = perceptron model
# Use a sigmoid function, we get a logistic regression model
# The loss function in stochastic is a Logarithmic loss unlike logistic regression loss
# if there is one dep has a binary outcome, loss is 'binary_crossentropy', if it has more than 2 outcomes it is called 'categoriacal_crossentropy'.
# Accuracy criteria to evaluate the observation weights to imporve models performace.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Use the fit method to fit the ANN to the training set
# change values of batch size to implement either reinforced learning or batch learning.
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
# Predict method gives probabilities, confusion matrix fives the values as T or F
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# This to convert values, you consider a 50% threshold.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)