import glob
import cv2
import gc
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

image_dir = './data/train/'

# Function to read image as RGB
def read_image(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_label(digit):
    label = [0.0 for i in range(11)]
    label[int(digit)]=1.0
    return np.array(label)
    #return label

# Batch generator that loads images in batches batch size is 4*batch_size
def batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        # Shuffle at the start of  every epoch
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset: offset+batch_size]
            # Load left right and centre images
            images = []
            labels = []
            for batch_sample in batch_samples:
                image = read_image(batch_sample)
                #print(batch_sample)
                digits = batch_sample.split("/")[-1][21:25]
                #print(digits)
                label = [generate_label(digits[0]),generate_label(digits[1]),generate_label(digits[2]),generate_label(digits[3])]
                images.append(image)
                labels.append(label)
            images, labels = shuffle(images,labels)
            x_train = np.array(images)
            labels = np.array(labels)
            y_train = [labels[:,0],labels[:,1],labels[:,2],labels[:,3]]
            gc.collect() # Run garbage collection to remove the leaky memory
            yield (x_train , y_train)
# from glob import glob
# samples = glob("./data/train/*.png")
# train_generator = batch_generator(samples, batch_size=64)
# x,y=next(train_generator)
# print(y[0])
