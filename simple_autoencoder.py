# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:10:30 2019

@author: leoska
"""

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, RepeatVector, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
import scipy.io as sio
import random
from dataset import create_dataset
from subplots import plot_history, plot_examples, network_evaluation
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SimpleAutoEncoder:
    def __init__(self, encoding_dim = [64], signal_len = 481):
        self.encoding_dim = encoding_dim
        self.signal_len = signal_len
        self.encoder = None
        self.decoder = None
        self.model = None
        
    def _encoder(self):
        input_window = Input(shape=(self.signal_len,))
        x = Dense(self.encoding_dim[0])(input_window)
        
        # "encoded" is the encoded representation of the input
        encoded = LeakyReLU(alpha=0.2)(x)
        
        # this model maps an input to its encoded representation
        encoder = Model(inputs = input_window, outputs = encoded)
        
        self.encoder = encoder
        return encoder
    
    def _decoder(self):
        encoded_window = Input(shape=(self.encoding_dim[0],))
        x = Dense(self.signal_len)(encoded_window)
        
        # "decoded" is the lossy reconstruction of the input
        decoded = LeakyReLU(alpha=0.2)(x)
        
        # create the decoder model
        decoder = Model(inputs = encoded_window, outputs = decoded)
        
        self.decoder = decoder
        return decoder
    
    def autoencoder(self):
        # Encoder model
        ec = self._encoder()
        
        # Decoder model
        dc = self._decoder()
        
        
        
        
        

#%%
# длина сигнала
signal_len = 481
# количество сигналов
signal_count = 474
# количество каналов
inChannel = 1

# Hidden size (размер сжатого слоя)
encoding_dim = 64 
# эпох обучения
epochs = 30 
batch_size = 64
# кол-во классов
num_classes = 10

#%%
tensorboard = TensorBoard(log_dir='tb_logs', histogram_freq=1, write_graph=True)
callback_list = [tensorboard]
print("TensorBoard and CallBacks init successful")

#%%
files_path = "data_emg/" #path to your directory with 10 data.txt files
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)

print("Load data successful")

#%%
print("Train signal shape: " + str(train_signals.shape))
print("Train labels shape: " + str(train_labels.shape))
print("Test signal shape: " + str(test_signals.shape))
print("Test labels shape: " + str(test_labels.shape))
print("Val signal shape: " + str(val_signals.shape))
print("Val labels shape: " + str(val_labels.shape))

x_train_simple = np.asarray(train_signals).reshape((len(train_signals), np.prod(np.asarray(train_signals).shape[1:])))
x_test_simple = np.asarray(test_signals).reshape((len(test_signals), np.prod(np.asarray(test_signals).shape[1:])))

#print(x_train_simple.shape)
print(np.asarray(x_train_simple).shape)
print(np.asarray(x_test_simple).shape)

plt.figure(figsize=(15,7))
plt.plot(x_train_simple[3] , 'r', label="Тренировочный сигнал")
plt.plot(x_test_simple[3] , 'b', label="Тестовый сигнал")
plt.title("Пример входного сигнала")
plt.legend()
plt.show()

#%%
# this is our input placeholder
model = Sequential()
model.add(Input(shape=(signal_len,)))
model.add(Dense(encoding_dim))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(signal_len))
model.add(LeakyReLU(alpha=0.2))

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

history = model.fit(x_train_simple, x_train_simple,
                steps_per_epoch=10,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test_simple, x_test_simple),
                validation_steps = 10,
                verbose=1,
                callbacks=[tensorboard])

decoded_stocks = model.predict(x_test_simple)

#%%
print(decoded_stocks.shape)
print(history.history.keys())

plot_history(history)
plot_examples(x_test_simple, decoded_stocks)
network_evaluation(history, epochs, batch_size)