# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:15:48 2018

@author: lucky
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset.
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1 ].values
y = dataset.iloc[:, -1].values

# While training the network we only use the value of X and not y cause 
# Unsupervised learning we do not consider dependant variable

# Feature scaling: Since there are high computations to be made
# Cause the dataset has lots of non linear relationships, it will be much easier for
# the network to be trained.

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
# There are various implementations that you can use.
# Here we are using Minisom. For working it is important to have Minisom.py file in your directory.

from minisom import MiniSom
# x, y are grid size of the som
# input_len = no of features in X = 14 attributes and cust ID.
# We actually do not need customer ID but we can use it to identify it.
# Sigma is the radius of the different neighborhoods in the grid.
# Learning_rate = the higher, the faster the convergence, default = 0.5.
# Decay function can be used to improve convergence.
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Initialize the weights input data on which model is trained.
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
'''
Here we are plotting the SOM itself in  a 2 by 2 grid which will contain all the final winning nodes
And for each of this nodes we will get IMPORTANT - MiD - Mean Interneuron Distance.
MiD of a specific winning node - is the mean of the distances of all the neurons around the winning node neighborhood.
The higher the MiD then the winning node is farther than its neighbors.
The outlier is farther away from rest of the nodes groups
'''

from pylab import bone, pcolor, colorbar, plot, show

# Bone function is used to make a window to display
bone()

# Put the different winning nodes on the map. Add the information of the MiD of all the winning
# nodes that the SOM identified. Diff colors correspond to the MiD range values. use pcolor for this
pcolor(som.distance_map().T)

# It is best to know what the colors stand for. for this we add the legend feature.
colorbar()

# The Frauds can be identified from the outlying winning nodes as they are far from other clusters of neighboring neurons.
# The Dark colors MiD is pretty low so are dense and close together.

# We are going to segregte customers basing on if they got approval or not fraudulantly.
# Here the Red circles - Customers who didnt get approval and Green squares - customers who got approval.

markers = ['o', 's']
colors = ['r', 'g']

# We loop over all the customers and for each customer we will get the winning node
# Depending on if the customer got approval or not - Red circle if the customer didnt get approval or green if he did. 

# In the loop we will be needing 2 variablels where i reps each customer and x is the corresponding cust vector.
# We use Enumerate of X as it contains all the customers.
# At makers[y[i]] we associate the value of markers with the value of y ie if approved or not. Therefore determining the value of markers.
for i, x in enumerate(X):
	winning_node = som.winner(x)
	plot(winning_node[0] + 0.5,
		  winning_node[1] + 0.5,
		  markers[y[i]],
		  markeredgecolor = colors[y[i]],
		  markerfacecolor = 'None',
		  markersize = 10,
		  markeredgewidth = 2)
show()  

'''
Observation - In the areas where there is a high risk of fraud there are both red and green areas which indicate that the
fraud customer applied and got denied but there are customers who got away with it.
'''

# Catch the cheaters
# Get the explicit list of all the customers who might have committed a fraud.
# Win_some takes an argument on which the SOM was generated
mappings = som.win_map(X)

# For determining the frauds, we use the mapping dictonary and enter the coordinated from the map generated.
# You can also use the concatenate function using np.concatenated
# example np.concatente((mappings[(5, 1)], mappings[(6, 8)]), axis = 0)
# Axis indicates that the values to be added vertically axis = 0;
frauds_data = np.concatenate((mappings[(5, 1)], mappings[(2, 2)]), axis = 0)

# Use the inverse transform to get back the original values
frauds_data = sc.inverse_transform(frauds_data)
