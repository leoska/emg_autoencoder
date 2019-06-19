# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:05:32 2019

@author: leoska
"""

from convautoencoder import ConvAutoEncoder
from dataset import create_dataset
from subplots import plot_signal, plot_signals, plot_examples
from tensorflow.keras import backend as K

#%%
files_path = "data_emg/" #path to your directory with 10 data.txt files
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)

train_signals = train_signals[:, 1:, :]
val_signals = val_signals[:, 1:, :]
test_signals = test_signals[:, 1:, :]

print("Train signal shape: " + str(train_signals.shape))
print("Train labels shape: " + str(train_labels.shape))
print("Test signal shape: " + str(test_signals.shape))
print("Test labels shape: " + str(test_labels.shape))
print("Val signal shape: " + str(val_signals.shape))
print("Val labels shape: " + str(val_labels.shape))

print("Load data successful")

#%%
plot_signal(train_signals[0], label = "Тренировочный сигнал", title = "Пример сигнала")

plot_signals(signals = [train_signals[3], test_signals[3]], colors = ['r', 'b'],
             labels = ["Тренировочный сигнал", "Тестовый сигнал"], title = "Сравнение сигналов")

#%%
convAutoEncoder = ConvAutoEncoder(signal_len = 480)
model = convAutoEncoder.autoencoder()
history = convAutoEncoder.fit(train_data = train_signals, validation_data = [val_signals, val_signals])

decoded_stocks = model.predict(test_signals)

#%%
score = convAutoEncoder.evaluate(test_data = test_signals)
print(score)
print('Test accuracy:', score[1])

#%%
print(decoded_stocks.shape)
print(history.history.keys())

convAutoEncoder.plot_history()
convAutoEncoder.network_evaluation()

plot_examples(test_signals, decoded_stocks)

#%%
# Закодированные сигналы
encoded_data = convAutoEncoder.encoder.predict(test_signals)
plot_examples(test_signals, encoded_data)
print(encoded_data[1].shape)

#%%
convAutoEncoder.savemodel()

#%%
K.clear_session()