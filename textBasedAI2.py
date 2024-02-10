#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:48:57 2024

@author: amirali
padding 
transformers

"""

from keras.datasets import reuters
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt



(train_data, train_targets), (test_data, test_targets) = reuters.load_data()

data = np.zeros((len(train_data), 2376))
for i, sublist in enumerate(train_data):
    data[i] = np.concatenate([np.array(sublist), np.zeros(2376 - len(sublist))])


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))  # 46 output classes
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 4 # Number of folds
print(k)
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate( [train_data[:i * num_val_samples], train_data[(i + 1)
    * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=1, verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)
    

all_mae_histories = []
all_mae_historiesT = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1)
    * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
    validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    mae_historyT = history.history['loss']
    all_mae_histories.append(mae_history)
    all_mae_historiesT.append(mae_historyT)
    

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_mae_historyT = [np.mean([x[i] for x in all_mae_historiesT]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history, 'r', label = 'validation loss')
plt.plot(range(1, len(average_mae_historyT) + 1), average_mae_historyT, 'b', label =
'training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
