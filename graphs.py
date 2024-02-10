from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
np.random.seed(1671)


NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
#OPTIMIZER = SGD()
#OPTIMIZER = RMSprop() # uses momentum
OPTIMIZER = Adam() # uses momentum 
N_HIDDEN = 128
VALIDATION_SPLIT=0.2
DROPOUT = 0.3

(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = to_categorical(y_train, NB_CLASSES)
Y_test = to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
#----------------------------------
# This 3 lines can be considered a hidden layer which uses the output of the previous 3 lines (layer) as it's input 
# That is why it doesn't need the input_shape=(RESHAPE,) as the initial input 
model.add(Dense(N_HIDDEN))
model.add(Activation('softmax'))
model.add(Dropout(DROPOUT))
#----------------------------------
#---------------------------------------
# Here we make some changes to the layer like the number of levels neurons within each layer to 254
# Change activation function to sigmoid
# Dropout radios to 0.5 
model.add(Dense(254))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
#---------------------------------------------
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])






history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluating the model on the test set
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])