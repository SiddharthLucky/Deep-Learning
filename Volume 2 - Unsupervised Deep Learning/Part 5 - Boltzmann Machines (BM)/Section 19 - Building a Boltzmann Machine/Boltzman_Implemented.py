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
test_set = torch.FloatTensor(test_set)

# Convert ratings into binary ratings
#here we will replace the zeros are not existant in the table, so they will be changed to -1, -1  is not a specific 
# Observation: We replace all the zeros with -1, the zeros in the training set are the ones that are not rated by the users,
# Since the ratings can only be 0 or 1, all the ratings with -1 correspond to the ratings that were not rated by the users.

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
# With this all the reatings with ratings - will be changed into -1
# All the values that are more than3 or above wil be made 1

# Building the architecture of the RBM
# REMEMBER: An RBM is a probabilistic grpaphical model
# Functions  defined 1, to initialize the RBM object 2, Sample the hidden nodes given the visible nodes 3, It will sample the probablities of the hidden nodes given.
# The name of the class 1st letter should be capitalized.
class RBM():
    # self corresponds to the object itself
    def __init__(self, no_of_VisNodes, no_of_HidNodes):
        # We initialize the paramenters of the future objects specific to the RBM class
        # Since these are variables of the object, you will have to use self as they dont pertain to the global variables.
        self.Weights = torch.randn(no_of_HidNodes, no_of_VisNodes)
        # Note: There is 1 bias for each hidden node, we need to create a vecotr of no_ofhidden nodes elements
        # Here we will make a vector with 2 args which are for the batch sizes and the bias itself, we add a fake vector cause pytorch needs 2 vectors for input
        ## Creating bias for hideen nodes
        self.a_hid = torch.randn(1, no_of_HidNodes)
        # Creating bias for visible nodes, 2D tensor
        self.b_vis = torch.randn(1, no_of_VisNodes)
    
    # This function will return sample of hidden nodes, it will sample the activations of the hidden nodes
    # if there are ex 100 hidden nodes, it will activate them according to a particular probability with an activation funtion.    
    # Since we declared the variables as self we will be able to access the above variables.
    # These are functions ex: sample_h is a sampling function for the hidden nodes.
    def sample_h(self, x):
    # In this x will correspond to the visible nuerons in the probability P(h) given V.
    # First we compute the probabilty of P(h) give V.
    # if the batch value contains 1 input vector of bias, those are called a mini batch
        wx = torch.mm(x, self.Weights.t())
        activation = wx + self.a_hid.expand_as(wx)
    # The activation function that will activate the hidden nodem, reppresents the probability the hidden nodes will be activated.
    ####### case Scenario #####
#    if there is a user who likes drama movies, if there is ahidden node that detected a feature corresponding to drama movies:
#    for this user who has high rating for drama movies, the probability specific to the node of drama will be very high.
#    as P(h) coreesponds to the drama feature and the V represents the user who likes drama.    
        
    ###### Case SCenario End#####
        P_Hid_given_Vis = torch.sigmoid(activation)
        # Function for gibbs sampling
        # Here we are make a burnoulli birnary RBM where in it only return if the user likes or dislikes the movie.
        return P_Hid_given_Vis, torch.bernoulli(P_Hid_given_Vis)
        
    # Measure the probabilities of the visible nodes.
    # It will be similar to all the notes made above but in place of hidden you are referring to the visible nodes.
    # They interchange places with each other.
    
    # This is a sample funtion that return the values of the visible nodes given the values of the hidden nodes.
    def sample_v(self, y):
        # Important: Here we dont take a transpose as we are computing P(v) given h, previously it was the other way round
        wy = torch.mm(y, self.Weights)
        activation = wy + self.b_vis.expand_as(wy)
        P_Vis_given_Hid = torch.sigmoid(activation)
        # Here we know if the value is 70 or above will be a lo=ike but any less than 25 is a dislike
        return P_Vis_given_Hid, torch.bernoulli(P_Vis_given_Hid)
    
    # Function for contrastive divergence which will approximate likelihood gradience or loglikelihood, we use gradient approximations.
    # In This we are trying to get to the lowest energy point.
    # Arguments explained
    
    # v0:  input vector containing ratings of all the movies by 1 user
    # vk: Visible nodes obtained after k roundtrips from the Hidden to visible and vise versa
    # ph0: Vector of probabilities that at the first iterations the hidden nodes values = 1 gibne the values of v0.
    # phk: probabilities of k sampling given the values of visible nodes vk
    
    def train(self, v0,  vk, ph0, phk):
        # Algorithm is followed as in the paper.
        self.Weights += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b_vis += torch.sum((v0 - vk), 0)
        self.a_hid += torch.sum((ph0 - phk), 0)
    # This will be used to perform contrastive divergence using gibbs sampling
    
# here we have to create an object to access the class,
# The value of no_of_VisNodes can be taken a value of nb_movies, but the right way to assign this would be
# To use the value of the torch tensor

no_of_VisNodes = len(training_set[0])
# Here the no of hidden nodes correponds to the no of features we want to detect
# That means no of stars, if oscar given and stuff will be used to determine the hidden node layers. 
no_of_HidNodes = 100 # Hard to say what us the optimum no of features. this parameter is totally tuneable
# You can also used to tune the value of batch size.
batch_size = 10 # for faster training you can tune the batch sizes for better optimization.
rbm = RBM(no_of_VisNodes, no_of_HidNodes);

# Training of the RBM
nb_epochs = 30
for epoch in range(1, nb_epochs + 1):
    # Most common loss function that is generally used: RMAC = Root of the mean of the squared differences between the predicted and real ratings.
    # You can also use the simple absolute difference between the two nodes predicted and real ratings
    train_loss = 0
    s = 0.
    # this is used as a counter increantal float value
    # The above functions are designed to work for ony one user,
    # So another for loop is made to get the values of the users into it, all the users in a batch.
    # here in the loop we dont want it to update after each user but we want it to do it in a batch
    for id_user in range(0, nb_users - batch_size, batch_size):
        # Will be output of the Gibbs sampling, input batch of the ratings of users 
        vk = training_set[id_user:id_user+batch_size]
        # Target is one thing we dont want to tamper. Here we use it to compare with the predicted with the orginal ratings
        v0 = training_set[id_user:id_user+batch_size]
        # Include the initial probabilities, ph is 1 given the real rating by the users.
        # We use sample_h as we 
        # By using a ,_ python understands that we only nned the first return argument
        # We use sample_h as we give in the values of V to p(h)
        ph0, _ = rbm.sample_h(v0)
        # Here we use another for loop for the k steps of contrastive divergence.
        for k in range(10):
            # Gibbs sampling is making several gibbs chains which are several round trips from the hidden to visible nodes and vice versa.
            #
            # We start with an input batch of operations - all the input ratings that are in v0 - batch of 100 users.
            # We get all the ratings of all the users from all the movies, then in the first ste of gibbs sampling
            # Step 1 - from the input vector we are going to sample the hidden nodes using a bernoulli sampling along with p_Hid_given_Vis 
            #
            # We are using sample_h function to get the first sampled hidden nodes, we use the same inverse trick.
            # Here hk is the hidden nodes obtained at the Kth step of contrastive divergence.
            # We use the variable Vk instead of V0 as we dont want to tamper with the original set.
            _,hk = rbm.sample_h(vk)
            # Vk is the sampled visible nodes after the first step of gibbs sampling
            # the below statement will generate the first sample of hidden nodes.
            _,vk = rbm.sample_v(hk)
            # Here what is happening is we get a first update of visible nodes, when k = 1
            # it enters the loop Since vk has the no of visible nodes and is passed to sample_h
            # Similarly the obtained output is then passed as an input to another function 
            # now after this we can approximate the gradients
            # We can use the value vk to approximate the gradient and update the bias.
            # WE ALSO HAVE TO MAKE SURE THAT WE DONT LEARN WHERE THERE IS NO RATING SPECIFIED. IE RATINGS = -1.
            vk[v0 < 0] = v0[v0 < 0]
            # if you look at the above step closely you can see that the machine is not learning when the ratings values are -1.
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        # Here when the weights get close to the optimal weights, the train loss is updated to see the error rate
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: ' + str(epoch) + 'loss: ' + str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s_test = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1] # vt is the target
    # Accoring to Markov Chain Monty Carlo sampling we are trained to stay in th eline blind folded and follow line in 10 steps
    # So it is possible to use the same training for staying on line for 1 step.
    if len(vt[vt >= 0]):
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # Here using vt we get the indexes of the cells that have the existant ratings
        s_test += 1.
print('test loss: ' + str(test_loss/s_test))


# Eva
    

