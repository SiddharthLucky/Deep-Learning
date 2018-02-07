# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

import tensorflow

# Importing the Keras libraries and packages
from keras.models import Sequential   # Initialize the Neural network
from keras.layers import Conv2D       # Covolution layer addition
from keras.layers import MaxPooling2D # Add pooling layers
from keras.layers import Flatten      # Convert all feature maps into feature vector for input vector
from keras.layers import Dense        # Add fully connected layers in the Cnn

# Initialising the CNN
classifier = Sequential()

# Step 1 - Adding the Convolution layer
# Conv2D parameters include the total number of feature detectors to be applied, row, column of features.
# input shape we specify the expected format for input images, converted into a 2D array. color into 3D images
# Here 3 means that the image is in color.
# Tensorflow arguments are theanos arguments in reverse.
# if it is a b/w image it would be one argument.
# relu is used to make sure there are no negetive values on the feature maps.
# If there are any negetive value pixels, remove them inorder to have non linearity, classifying is a non linear problem. 
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Adding the Pooling layer
# Used to reduce the sizs of the feature map. Check online, check max pooling. Reduce the size of feature maps to flatten.
# MP is done by taking the maximum of feature maps. We dont loose the performace and spacial structure.
# 2 by 2 is mostly the preferred size as we dont want to loose the features, this keeps the information and precise features of the input image.
# Recommended using that most of the time, cause it is good enough to take the high numbers and retain spacial features.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# We tahe the 2D maps and then convert them into a single vector.
# This will be fed into the ANN.
# If we directly input the pixel val of the image, then the spacial structure will become confusing.
# With reduced feature map
classifier.add(Flatten())

# Step 4 - Full connection
# Since it is tought to see no of inputs for estimation, you choosa numple power of 2 above 100 or so, tryp to optimize it more.
# Relu gives probabilities of actions.
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# Classification problems in general use Logarithmic loss.
# If more than 2 outcomes it would be categorical cross entropy.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# Image augmentation is preprocessing images to prevent overfitting.
# Overfitting sometimes happes when we have few data to train, but fails to generalize the correlations on new data.
# Data augmentation comes into play when we have a little less amounts of data. It will create many batches of images,
# Each batch will be rotated, shifted, augmented, to enrich available dataset to prevent overfitting.
# Image augmnetation will have multiple in keras documentation, check for the appropriate one.
# There are various image transformations that can be applied, listed on the keras documentation.
# trend generator andvalidation generator are used to create the test and training set.
classifier.fit()


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# you can use Alt + Shift to align the data according to python indentation
# class_mode can range from "categorical, sparse and binary"
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Arguments passed are from the names from above
# Validation data is the name of the test set
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)