# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:21:44 2020

@author: reine
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import time

# import dataset
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
#print(y_train)
#print(X_train[2])

# plot image
#plt.imshow(X_train[2])
#print(y_train[2])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#print(y_train)

X_train = X_train/255
X_test = X_test/255

# CNN MODEL
"""UNCOMMENT BELOW"""

cnn = Sequential()
cnn.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 20, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL FROM CLASS --')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy)


# CNN MODEL w/ 10 Epochs
cnn = Sequential()
cnn.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL WITH 10 EPOCHS --')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy)

# CNN MODEL w/ 2 Stages of Convolution Before Flattening 10 Epochs
cnn = Sequential()
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))

cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL WITH 2 STAGES OF CONVOLUTION BEFORE FOLDING --')
print('32 Filters First Convolution | 64 Filters Second Convolution | 3 Hidden Layers with 256 Neurons Per | 10 Epochs')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy)


# CNN MODEL w/ 2 Stages of Convolution and Dropout Before Flattening 10 Epochs
cnn = Sequential()
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.3)) #<-- Dropout Added
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(Dropout(0.3)) #<-- Dropout Added
cnn.add(MaxPooling2D(2,2))

cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL WITH 2 STAGES OF CONVOLUTION BEFORE FOLDING --')
print('32 Filters First Convolution | 64 Filters Second Convolution | 3 Hidden Layers with 256 Neurons Per | 10 Epochs')
print('Dropout 0.3 Added')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy)

# CNN MODEL w/ 3 Stages of Convolution and Dropout Before Flattening 50 Epochs
cnn = Sequential()
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), padding='same', activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 32, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Dropout(0.2)) #<-- Dropout Added

cnn.add(Conv2D(filters = 64, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Dropout(0.2)) #<-- Dropout Added

cnn.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Dropout(0.2)) #<-- Dropout Added

cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))

cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 50, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL WITH 3 STAGES OF CONVOLUTION BEFORE FOLDING --')
print('32 Filters First Convolution | 64 Filters Second Convolution | 128 Filters Second Convolution | 3 Hidden Layers with 256 Neurons Per | 50 Epochs')
print('Dropout 0.2 Added After Each Convolution')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy)


# MNIST DATA
from tensorflow.keras.datasets import mnist # <-- Import Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)

# reshape the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)


# plot image
#plt.imshow(X_train[2])

# change data type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the data
X_train = X_train/255
X_test = X_test/255

# onehot encode y data
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# build the model
cnn = Sequential()
cnn.add(Conv2D(filters = 8, kernel_size = (3,3), padding='same', activation = 'relu', input_shape = (28, 28, 1)))
cnn.add(Conv2D(filters = 8, kernel_size = (3,3), padding='same', activation = 'relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Dropout(0.2)) #<-- Dropout Added

cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 256, activation = 'relu'))

cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.time() # records initial start time
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 10, shuffle = True)
end = time.time() # records end time
loss, accuracy = cnn.evaluate(X_test, y_test)
seconds = end - start

print('-- CNN MODEL WITH 1 STAGE OF CONVOLUTION BEFORE FOLDING --')
print('32 Filters First Convolution 3 Hidden Layers with 256 Neurons Per | 10 Epochs')
print('Dropout 0.2 Added After Convolution')
print('Time to complete (seconds): ', seconds)
print('Time to complete (minutes):', seconds/60)
print('loss: ', loss)
print('accuracy: ', accuracy) 