# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:39:03 2019

@author: leoska
"""

from deepautoencoder import DeepAutoEncoder
from dataset import create_dataset
from subplots import plot_signal, plot_signals, plot_examples
from tensorflow.keras import backend as K
import numpy as np

#%%
files_path = "data_emg/" #path to your directory with 10 data.txt files
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)

print("Train signal shape: " + str(train_signals.shape))
print("Train labels shape: " + str(train_labels.shape))
print("Test signal shape: " + str(test_signals.shape))
print("Test labels shape: " + str(test_labels.shape))
print("Val signal shape: " + str(val_signals.shape))
print("Val labels shape: " + str(val_labels.shape))

print("Load data successful")

#%%
plot_signal(train_signals[0], label = "Тренировочный сигнал", title = "Пример сигнала")

x_train_deep = np.asarray(train_signals).reshape((len(train_signals), np.prod(np.asarray(train_signals).shape[1:])))
x_test_deep = np.asarray(test_signals).reshape((len(test_signals), np.prod(np.asarray(test_signals).shape[1:])))
x_val_deep = np.asarray(val_signals).reshape((len(val_signals), np.prod(np.asarray(val_signals).shape[1:])))

print("x_train_deep shape: " + str(np.asarray(x_train_deep).shape))
print("x_test_deep shape: " + str(np.asarray(x_test_deep).shape))
print("x_val_deep shape: " + str(np.asarray(x_val_deep).shape))

plot_signals(signals = [x_train_deep[3], x_test_deep[3]], colors = ['r', 'b'],
             labels = ["Тренировочный сигнал", "Тестовый сигнал"], title = "Сравнение сигналов")

#%%
deepAutoEncoder = DeepAutoEncoder()
model = deepAutoEncoder.autoencoder()
history = deepAutoEncoder.fit(train_data = x_train_deep, validation_data = [x_val_deep, x_val_deep])

decoded_stocks = model.predict(x_test_deep)

#%%
score = deepAutoEncoder.evaluate(test_data = x_test_deep)
print(score)
print('Test accuracy:', score[1])

#%%
print(decoded_stocks.shape)
print(history.history.keys())

deepAutoEncoder.plot_history()
deepAutoEncoder.network_evaluation()

plot_examples(x_test_deep, decoded_stocks)

#%%
# Закодированные сигналы
encoded_data = deepAutoEncoder.encoder.predict(x_test_deep)
plot_examples(x_test_deep, encoded_data, colors = ['r', 'g'])

#%%
K.clear_session()