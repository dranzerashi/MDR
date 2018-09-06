from keras.layers import Input, Dense
from keras.models import Model
import numpy 
numpy.random.seed(7)
dataset = numpy.loadtxt("dataset.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:10,0:8]
Y = dataset[:10,8]
# This returns a tensor
inputs = Input(shape=(8,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(1024, activation='relu')(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions_1 = Dense(2, activation='softmax')(x)
predictions_2 = Dense(1, activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=[predictions_1, predictions_2])
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
model.fit(X, [numpy.stack([Y,Y],axis=1),Y],epochs=150, batch_size=10) # starts training

scores = model.evaluate(X, [numpy.stack([Y,Y],axis=1),Y])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# round predictions
print(predictions)