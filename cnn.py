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
model_m = Sequential()
model_m.add(Reshape((481, 1), input_shape=(train_signals.shape[1], train_signals.shape[2])))
model_m.add(Conv1D(50, 10, activation='relu', input_shape=(481, 1)))
model_m.add(Conv1D(25, 10, activation='relu'))
model_m.add(MaxPooling1D(4))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(Conv1D(50, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(10, activation='softmax'))
print(model_m.summary())

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

history = model_m.fit(train_signals, train_labels,
                      steps_per_epoch=10,
                      epochs=20,
                      batch_size=64,
                      validation_data=(val_signals, val_labels),
                      validation_steps=10,
                      verbose=0
)

#model_m.save(files_path + "model.h5")

#%%
loss, accuracy = model_m.evaluate(x = test_signals, y = test_labels, batch_size=64) # evaluating model on test data
print("test loss: ", loss)
print('Test accuracy:', accuracy)

plot_history(history)
network_evaluation(history=history, epochs=20, batch_size=64)

#%%
print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(test_signals)
# Take the class with the highest probability from the test predictions
y_test = np.asarray(y_pred_test).reshape((len(y_pred_test), np.prod(np.asarray(y_pred_test).shape[1:])))
max_y_pred_test = np.argmax(y_test, axis=1)
max_y_test = np.argmax(test_labels, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)