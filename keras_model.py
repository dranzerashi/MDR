import numpy as np
import scipy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPool2D, Flatten, Input


class Model(object):
    """CNN architecture:
       INPUT -> CONV -> RELU -> CONV -> RELU ->
       POOL -> CONV -> POOL -> FC -> RELU -> 5X SOFTMAX
    """
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.model = Sequential()
        model = self.model

        # First convolutional layer
        # 16 filters - size(5x5x3)
        model.add(Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        #skipping local response normalization
        model.add(Dropout(self.keep_prob))
        
        # Second convolutional layer
        # 32 filters - size(5x5x16)
        model.add(Conv2D(filters=32,kernel_size=5, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        #skipping local response normalization
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Dropout(self.keep_prob))

        # Third convolutional layer
        # 64 filters - size(5x5x32)
        model.add(Conv2D(filters=64,kernel_size=5, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        #skipping local response normalization
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        
        # Reshape tensor from POOL layer for connection with FC
        model.add(Flatten())
        model.add(Dropout(self.keep_prob))

        # Fully connected layer
        model(Dense(1024, activation='relu'))
        model.add(Dropout(self.keep_prob))

        # Create variables for 5 softmax classifiers
