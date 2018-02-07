# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:39:15 2017

@author: lucky
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#Here the upper bound is excluded.
X = dataset.iloc[:, 3 : 13].values
#y is the output to the dependent variable
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000],[0.0, 0, 619, 1, 42, 3, 58000, 2, 1, 1, 70000]])))
final_prediction = (new_prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating an ANN
# Kfold belongs to scikit learn
# Wrapper in keras to wrap kfold and ANN model ie keras classifier

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# It will build the classifier the same way as above create everything apart from fitting
def build_classifier():
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

# This will not be trained using the same x_train and y_train validation
# But will be trained using K fold validation eachtime measuring performance on 1 test fold
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
# Most of the time 10 folds are used.
# n_jobs is the no of CPU's to be used.
# The batch_size indiicates after how many samples the gradient descent is updated
'''
We are running this for 2 reasons.
1. if the real accuracy is near to the first or second accuracy.
2. to check in what bias and variance we are at.
3. Set the value of n_jobs to -1 so that thecomputations run in parellel
'''

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Calculating the mean and the variance 
mean = accuracies.mean()
vairance = accuracies.std()

	