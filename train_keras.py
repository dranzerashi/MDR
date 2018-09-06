import numpy as np
import tensorflow as tf
from preprocess_keras import batch_generator 
from keras_model import MDRModel
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




def predictions(logit_1, logit_2, logit_3, logit_4):
    """Converts predictions into understandable format.
    For example correct prediction for 2 will be > [2,10,10,10,10]
    """
    first_digits = np.argmax(logit_1, axis=1)
    second_digits = np.argmax(logit_2, axis=1)
    third_digits = np.argmax(logit_3, axis=1)
    fourth_digits = np.argmax(logit_4, axis=1)
    
    stacked_digits = np.vstack((first_digits, second_digits, third_digits, fourth_digits))
    rotated_digits = np.rot90(stacked_digits)[::-1]
    return rotated_digits


def accuracy(logit_1, logit_2, logit_3, logit_4):
    """Computes accuracy"""
    correct_prediction = []
    y_ = y_[:, 1:5]
    rotated_digits = predictions(logit_1, logit_2, logit_3, logit_4)
    for e in range(len(y_)):
        if np.array_equal(rotated_digits[e], y_[e]):
            correct_prediction.append(True)
        else:
            correct_prediction.append(False)       
    return (np.mean(correct_prediction))*100.0        


def train(reuse, batch_size=256, nb_epoch=100000):
    """Trains CNN."""
    samples = glob("./data/train/*.png")
    
    # Split the data into 80:20 training and test samples
    training_samples, validation_samples = train_test_split(samples, test_size=0.2)

    
    train_generator = batch_generator(training_samples, batch_size=batch_size)
    validation_generator = batch_generator(validation_samples, batch_size=batch_size)
    
    print("Data uploaded!")
    model = MDRModel(0.5)
    model.model.fit_generator(train_generator, steps_per_epoch=len(training_samples)/batch_size, validation_data=validation_generator, nb_val_samples=len(validation_samples)/batch_size, epochs=500)

    #history = model.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))

if __name__ == '__main__':
    train(reuse=False)
