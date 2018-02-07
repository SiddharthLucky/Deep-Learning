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

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#encoder for changing strings to numbers.
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
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

# Fitting classifier to the Training set

# Importing Keras library and modules for building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initializing the ANN
classifier = Sequential()

# Adding the first input layer and hidden layer with dropout
classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Do not go with more than 0.5 it will cause underfitting
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(p = 0.1))
 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000],[0.0, 0, 619, 1, 42, 3, 58000, 2, 1, 1, 70000]])))
final_prediction = (new_prediction > 0.5)

# This to convert values, you consider a 50% threshold.
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
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# Most of the time 10 folds are used.
# n_jobs is the no of CPU's to be used.
# The batch_size indiicates after how many values the batch should be updated
'''
We are running this for 2 reasons.
1. if the real accuracy is near to the first or second accuracy.
2. to check in what bias and variance we are at.
3. Set the value of n_jobs to -1 so that thecomputations run in parellel
'''
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

# Calculating the mean and the variance 
mean = accuracies.mean()
vairance = accuracies.std()

# Imporving an ANN
# Dropout regularization, is the solution to overfitting.
# Overfitting means if your model is trained too much over a training set, it becomes less performing on the test set
# Overfitting happens when the accuracy of the training set is higher and the acc on test set is lower,
# You can also detect high variance when applying kFold cross validation, youre model learned too much.

'''
Dropout regularization is applied by disabling a couple of neurons after each iteration so that they learn 
to be independant than to depend on another neurons corelation.
This way many new independent configurations can be found.

'''

'''
PARAMETER TUNING

There are 2 types of parameters, that are learnt in training, there are parameters that are fixed called the hyper parameters.
These Hyperparameters could be the abtch_size or the no of epochs, the optimizer or the no of nuerons in a layer.
We might get to a better accuracy by changing them, best values of hyper parameters

This is performed by using kFold Cross validation,a technique called Grid Search
Will lead to the best choice of parameters.
'''
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# It will build the classifier the same way as above create everything apart from fitting
def build_classifier(optimizerarg):
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = optimizerarg, loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

# This will not be trained using the same x_train and y_train validation
# But will be trained using K fold validation eachtime measuring performance on 1 test fold
	
# In this we will not input the epoch size and batch size as they have to be tuned, put seperately
'''
If you want to iterate the optimizer value, you can pass the name of the optimizer as an argument,
check optimizerarg, it is recommended that you use RMSprop for recurrent neural networks.
You can also dynamically define variables to iterate overtime, reducing time taken to manually change values.
'''
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
			  'epochs': [100, 500],
			  'optimizerarg': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)

# Fitting the grid search to the ANN, training set
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_Accuracy = grid_search.best_score_


