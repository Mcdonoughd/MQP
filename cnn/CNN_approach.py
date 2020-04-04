#!/usr/bin/env python
# coding: utf-8

# CNN approach to classify images. Starting with CIFAR10 w/ 10 classes of things from 50000 train and 10000 test

# In[ ]:


#create mapping from 0-2 to represent the strings instead of their integers.
num_name = ['healthy',
            'damaged',
            'recovering']


# Determine the x_train/test and y_train/test where:
# 
# 
# *   x_train is the labeled training data, and test is what you're testing on
# *   y_train is the labels mapping to x_train
# 
# 

# In[ ]:



#x_train is the labeled training dataset of images.
x_train = ...
#y_train should be a one-dimensional list of the labels in the [0,1,2] category for each image corresponding in x_train
y_train = ...
#the testing dataset, unlabeled
x_test = ...
#labels of testing dataset
y_test = ...


# In[ ]:


import matplotlib.pyplot as plt
#just show one of the images
img = plt.imshow(x_train[2])


# In[ ]:


#print that cell's class.
print(num_name[int(y_train[2])])


# In[ ]:


#check the dimensions of each
print('xtrain', x_train.shape)
print('ytrain', y_train.shape)
print('xtest', x_test.shape)
print('ytest', y_test.shape)


# One Hot encoding to convert the labels into set of 3 values for NN input

# In[ ]:


from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Print new labels
print(y_train_one_hot[0])


# Normalize pixel values from 0 to 1 by dividing each pixel value by 255 (max pixel value).

# In[ ]:


x_train = x_train / 255
x_test = x_test / 255


# Build CNN using Sequential architecture
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#create architecture
model = Sequential()

#1st convolution layer to create feature maps
#using ReLU as is most common

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))

#MaxPooling2d layer to reduce dimensionality by half, for example, going to reduce 32x32 image to 16x16 - Is Maxpooling too far a concern for ours because the pixels are so important we don't want to reduce their dimensionality?
model.add(MaxPooling2D(pool_size=(2, 2)))

#Second convolution layer
model.add(Conv2D(64, (5, 5), activation='relu'))

#Second Maxpooling2d layer, reducing from 32-16-8
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten layer which makes the image into a 1d array for NN input
model.add(Flatten() )


#add neurons, using relu activation fn for first 1000 and softmax for 10
model.add( Dense(1000, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# Compile the model here
# As well as set loss function
#    * Categorical CrossEntropy = 74% across 10 epochs
#    * Hinge is bad.
# going to try both and evaluate how it tests.
# 
# Set optimizer function, going to use Adam over stochastic gradient descent

# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# Time to train/fit the model
# Batch size = total # of training examples.
# 43s for 10 epochs 84% GPU
# 

# In[ ]:


import time

start = time.time()
hist = model.fit(x_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.3 )
print(time.time() - start)


# Trying my own image to test this classifier.

# In[ ]:


# #Load the data -- may not work off of colab. irrelevant
# from google.colab import files # Use to load data on Google Colab
# uploaded = files.upload() # Use to load data on Google Colab


# In[ ]:


#load your own image here for testing.
my_image = plt.imread(...)


# In[ ]:


plt.imshow(my_image)


# need to rescale images to  32x32

# In[ ]:


#resizing the image from whatever their dimension was to the 32x32 size.
from skimage.transform import resize
resized = resize(my_image, (32,32,3))
img = plt.imshow(resized)


# Get probabilities for each of the 10 classes and store into var

# In[ ]:


import numpy as np
probabilities = model.predict(np.array([resized]))


# In[ ]:


probabilities


# In[ ]:


index = np.argsort(probabilities[0,:])


# In[ ]:


index


# In[ ]:


print("Most likely class:", num_name[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", num_name[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", num_name[index[7]], "-- Probability:", probabilities[0,index[7]])


# In[ ]:





# 
