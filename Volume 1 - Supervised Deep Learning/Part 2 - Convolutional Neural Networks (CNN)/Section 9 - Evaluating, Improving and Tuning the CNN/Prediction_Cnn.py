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
from keras.models import Sequential   
from keras.layers import Conv2D       
from keras.layers import MaxPooling2D 
from keras.layers import Flatten      
from keras.layers import Dense        

# Initialising the CNN
classifier = Sequential()

# Step 1 - Adding the Convolution layer
# In order to tune the network and increase accuracy,
# You can add anpther convelutional layer or add another Dense layer to make the network even deeper.
# This is all sheer experiment, no concrete results.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Adding the Pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# When adding this layer, you do not need to specify the input_shape as,
# the above Conv layer will be the input to the same.
# You can also increase the size of the feature detectors to 64 to look for more features.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# You can also add another fully connected layer in creasing the depth to reduce over fitting. 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.Dropout(rate = 0.1)
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
classifier.fit()


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# you can use Alt + Shift to align the data according to python indentation, check for windows.
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
# Validation data is the name of the test set.
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

'''
Making Single Predictions
'''

# Making sinigle image predictions
# Numpy is used to convert the image to a format nderstandable by the CNN.
import numpy as np
from keras.preprocessing import image

# The input image should kinda be the same size as the input_shape Attribute in the CNN.
test_image = image.load_img('subfolder/foldername/filename.jpg', target_size = (64, 64))

# The image from above to modify it from a 2D image to a 3D array
# Converted to a 3D format to fit the input_shape attribute in the CNN.
test_image = image.img_to_array(test_image)

'''
Keep in mind that when you initially run this with the test image as parameter it throws an error stating that
the array should be of 4 dimensions. 

A nueral network uses basic 4 dimesions in which the 4th dimesion is a batch. 
Networks dont accept single rows of data, but take them as a batch.
Here we are taking one element in a batch, butin general it can accept several batches of several inputs. 
'''
# Adding the extra dimms, value situated in the numpy package.
# Axis is to specify the index of the dimension we are adding. Ex gives the output (1, 64, 64, 3)
# For this we have to add a new dimension using expand Dimms.
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

# This gives the values of the indices that were fed into the network ex: {'cats: 0, 'dogs':1}
training_set.class_indices

# This condition can be used to directly specify what the output is:
if result[0][0] == 1:
	ult_prediction = 'dogs'
else:
	ult_prediction = 'cats'


