# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:58:05 2018

@author: lucky
"""

import numpy as np
import pandas as pd


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:3] 
#Select features in columns 1 and 2, change it to what is required
 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
 
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,:])
    y_train.append(training_set_scaled[i,0:1])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
