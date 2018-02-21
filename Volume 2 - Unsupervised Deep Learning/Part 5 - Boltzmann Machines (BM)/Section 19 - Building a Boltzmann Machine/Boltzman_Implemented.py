# -*- coding: utf-8 -*-

# Here the data preprocessing phases are the same.

###### Below is the information from the papaers.######
# This model contains contrastive divergence to estimate likelihood.

# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data 
from torch.autograd import Variable


# Importing the dataset
                ######################

#    Here the imports have to be augmented, include the directories too.

                ######################

# Do not use commas for seperator as movie names can contain the same so here we are using
# :: as seperator values. it can be anything but make sure the same date is not split into 2 orws and columns.   
# We do not use general encoding because of some of the special charecters in the titles.
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# 1. Breakdown of the tables ratings, in this the first column will be the movies, 
# 2, will be the movies no representing corresponsing movies, 
# 3, Ratings, corresponding values along with timestamps
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#####preparing training set and test sets#######
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
# For feeding pytorch the data is supposed to be of the format of an integer array.
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the total number of users
# The data will be saved as a matrix where lines are users and colunms are movies and cells are ratings
# 2 matrices for training and test sets, contains same no of lines and indexes, if no rating mentioned, pad it

# This would be for the max no of training and test for the set 1
# Similarly you will have to make them all in the same way
# in case you have any questions try to run the program, it will make most of it clear
nb_users  = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Sample scenario: we had to take the max of the max as the max could be anywhere, either in the training or the test set.
# try running max(training_set[:,0]) alone to see why running the max of both test cases is needed.

# RBM, type of neural networks expects the inputs nodes obs in lines and features in columns

# Function to convert the data in inputs for RBM

def convert(data):
    # Since we are using torch we are going to create a list of lists and not 2Darrays
    # We are using lists of lists where in each user is a list
    # Sampel description basin on problem, total of 943 lists cause of no of users and each user has ratings for 1682 movies
    # New_data is the value to be returned.
    new_data = []
    # Here the upper bound is excluded hence u add a + 1
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # Here we can use condition checking to select only users of particular ID, you can specify them in another sq braces.
        id_ratings = data[:, 2][data[:, 0] == id_users]
        # We are using 1682 values for zeros as if the user rated a movie the value will be 1 or else will be zero
        ratings = np.zeros(nb_movies)
        # Here we do an id_movies - 1 as the index starts from 0 in matrix ratings
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the matrix lists into Torch Tensors
# Tensors are arrays that contains elements of sinlge type.
# For these kind of problems tensprs are deemed less efficient
# here instead of using anumpy array, the data is converted into a float tensor
# Pytorch is fairly new so pytorch tensors might not show up. they exist but cant be displayed
training_set = torch.FloatTensor(training_set)
test_Set = torch.FloatTensor(test_set)

# Convert ratings into binary ratings