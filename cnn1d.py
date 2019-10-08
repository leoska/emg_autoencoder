import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

import tensorflow as tf
import numpy as np
import pickle
import time
from random import shuffle
from tensorflow import keras
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Reshape

#%%
from dataset import create_dataset
from subplots import plot_signal, plot_signals, plot_examples

files_path = "data_emg/" #path to your directory with 10 data.txt files
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)

#path to data files
path = r'./data_new/'

#path where you want to save trained model and some other files
sec_path = r'./pretrained_model/'

num_classes = 9
num_sensors = 1
input_size = train_signals.shape[1]

model = Sequential()
model.add(Reshape((input_size, num_sensors), input_shape=(input_size, )))
model.add(Conv1D(50, 10, activation='relu', input_shape=(input_size, num_sensors)))
model.add(Conv1D(25, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(100, 10, activation='relu'))
model.add(Conv1D(50, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Dropout(0.5))
#next layers will be retrained
model.add(Conv1D(100, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

start_time = time.time()

history = model.fit(train_signals, train_labels,
                      steps_per_epoch=25,
                      epochs=25,
                      batch_size=None,
                      validation_data=(val_signals, val_labels),
                      validation_steps=25
)

elapsed_time = time.time() - start_time # training time

loss, accuracy = model.evaluate(test_signals, test_labels) # evaluating model on test data

loss = float("{0:.3f}".format(loss))
accuracy = float("{0:.3f}".format(accuracy))
elapsed_time = float("{0:.3f}".format(elapsed_time))

#saving some data
f = open(sec_path + "info.txt", 'w')
f.writelines(["loss: ", str(loss), '\n', "accuracy: ", str(accuracy), '\n', "elapsed_time: ", str(elapsed_time), '\n'])

#saving model
model.save(sec_path + "pretrained_model.h5")

#saving test data just in case
cc = list(zip(test_signals, test_labels))
with open(sec_path + "pretrained_model_test_data.txt", "wb") as fp:
    pickle.dump(cc, fp)

#saving history
with open(sec_path + "pretrained_model_history.h5", "wb") as fp:
    pickle.dump(history.history, fp)

#%%
encoder = load_model(r'./weights/conv_encoder.h5')
encoded_data = encoder.predict(test_signals)
plot_examples(test_signals, encoded_data)

