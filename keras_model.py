import numpy as np
import scipy
import tensorflow as tf
from keras.models import Model, model_from_json
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

        # Second convolutional layer
        # 32 filters - size(5x5x16)
        x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        #skipping local response normalization
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(self.keep_prob)(x)

        # Third convolutional layer
        # 64 filters - size(5x5x32)
        x = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        #skipping local response normalization
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Reshape tensor from POOL layer for connection with FC
        x = Flatten()(x)
        x = Dropout(self.keep_prob)(x)

        # Fully connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.keep_prob)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.keep_prob)(x)

        # Create variables for 5 softmax classifiers
        self.d1 = Dense(11, activation='relu', kernel_initializer='glorot_uniform')(x)
        self.d2 = Dense(11, activation='relu', kernel_initializer='glorot_uniform')(x)
        self.d3 = Dense(11, activation='relu', kernel_initializer='glorot_uniform')(x)
        self.d4 = Dense(11, activation='relu', kernel_initializer='glorot_uniform')(x)
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = Model(inputs=input, outputs=[self.d1, self.d2, self.d3, self.d4])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def custom_loss(self):
        pass

    def save_model(self, file="model.h5"):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file)
        print("Saved model to disk")
    
    def load_model(self, file="model.h5"):
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        #loaded_model.load_weights("model.h5")
        self.model.load_weights(file)
        print("Loaded model from disk")