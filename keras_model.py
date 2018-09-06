import numpy as np
import scipy
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPool2D, Flatten, Input
from keras.optimizers import SGD


class MDRModel(object):
    """CNN architecture:
       INPUT -> CONV -> RELU -> CONV -> RELU ->
       POOL -> CONV -> POOL -> FC -> RELU -> 5X SOFTMAX
    """
    def __init__(self, keep_prob=1.0):
        self.keep_prob = keep_prob
        self.build_model()
    
    def build_model(self):
        input = Input(shape=(160, 196, 3))
        # First convolutional layer
        # 16 filters - size(5x5x3)
        x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same')(input)
        x = Activation('relu')(x)
        #skipping local response normalization
        x = Dropout(self.keep_prob)(x)

        # Second convolutional layer
        # 32 filters - size(5x5x16)
        x = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        #skipping local response normalization
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(self.keep_prob)(x)

        # Third convolutional layer
        # 64 filters - size(5x5x32)
        x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        #skipping local response normalization
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Reshape tensor from POOL layer for connection with FC
        x = Flatten()(x)
        x = Dropout(self.keep_prob)(x)

        # Fully connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.keep_prob)(x)

        # Create variables for 5 softmax classifiers
        self.d1 = Dense(11, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        self.d2 = Dense(11, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        self.d3 = Dense(11, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        self.d4 = Dense(11, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = Model(inputs=input, outputs=[self.d1, self.d2, self.d3, self.d4])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def custom_loss(self):
        pass

