"""Try to predict 7 random examples from test data"""

import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import preprocessing
from keras_model import MDRModel

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


x = np.array(glob("./data/train/*.png"))
indices = np.random.choice(len(x), 7)
examples = x[indices]
model = MDRModel()
model.load_model()
images = np.array([scipy.misc.imread(el, flatten=False) for el in examples])
preds = model.model.predict(images)
preds = predictions(*preds)


# Create matplotlib plot for visualization 
plt.rcParams['figure.figsize'] = (20.0, 20.0)
f, ax = plt.subplots(nrows=1, ncols=7)
for i, el in enumerate(images):
    ax[i].axis('off')
    number = preds[i][preds[i] < 10]
    ax[i].set_title("Pred: "+''.join(number.astype("str")), loc='center')
    ax[i].imshow(el)    
plt.show()
