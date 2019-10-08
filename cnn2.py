# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:08:54 2019

@author: leoska
"""

import os 
os.system('cls')
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Reshape, LeakyReLU

#%%
from dataset import create_dataset
from subplots import plot_history, network_evaluation, show_confusion_matrix

#%%
files_path = "data_emg/" #path to your directory with 10 data.txt files
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)

#%%
encoder = load_model(r'./weights/conv_encoder.h5')
train_signals = train_signals[:, 1:, :]
val_signals = val_signals[:, 1:, :]
test_signals = test_signals[:, 1:, :]   

train_signals_en = encoder.predict(train_signals)
val_signals_en = encoder.predict(val_signals)
test_signals_en = encoder.predict(test_signals)

#%%

model_e = Sequential()
model_e.add(Reshape((120, 1), input_shape=(train_signals_en.shape[1], train_signals_en.shape[2])))
model_e.add(Conv1D(50, 10, input_shape=(120, 1)))
model_e.add(LeakyReLU(alpha=0.2))
model_e.add(Conv1D(25, 10))
model_e.add(LeakyReLU(alpha=0.2))
model_e.add(MaxPooling1D(4))
model_e.add(Conv1D(100, 10))
model_e.add(LeakyReLU(alpha=0.2))
model_e.add(Conv1D(50, 10))
model_e.add(LeakyReLU(alpha=0.2))
model_e.add(GlobalAveragePooling1D())
model_e.add(Dropout(0.5))
model_e.add(Dense(10, activation='softmax'))
print(model_e.summary())

model_e.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

history = model_e.fit(train_signals_en, train_labels,
                      steps_per_epoch=10,
                      epochs=20,
                      batch_size=64,
                      validation_data=(val_signals_en, val_labels),
                      validation_steps=10)
#%%
loss, accuracy = model_e.evaluate(x = test_signals_en, y = test_labels,  batch_size=64) # evaluating model on test data
print("test loss2: ", loss)
print('Test accuracy2:', accuracy)

plot_history(history)
network_evaluation(history=history, epochs=20, batch_size=None)

#%%
print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_e.predict(test_signals_en)
# Take the class with the highest probability from the test predictions
y_test = np.asarray(y_pred_test).reshape((len(y_pred_test), np.prod(np.asarray(y_pred_test).shape[1:])))
max_y_pred_test = np.argmax(y_test, axis=1)
max_y_test = np.argmax(test_labels, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

#%%
print(train_signals.shape)