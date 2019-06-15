# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:03:37 2019

@author: leoska
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal, color = "b", label = "", title = ""):
    plt.figure(figsize=(15,7))
    plt.plot(signal, color, label = label)
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_signals(signals, colors, labels, title = ""):
    plt.figure(figsize=(15,7))
    for i in range(len(signals)):
        plt.plot(signals[i], colors[i], label=labels[i])
    plt.title(title)
    plt.legend()
    plt.show() 
        
# Функция, которая выводит метрики обучения нейронной сети
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    
# Функция выводит сравнение графиков
def plot_examples(stock_input, stock_decoded, test_samples = 158, step = 22):
    n = 10  
    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(list(np.arange(0, test_samples, step))):
        # display original
        ax = plt.subplot(4, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx], "r")
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
    plt.show()
        
def network_evaluation(history, epochs, batch_size):
    '''# оцениваем нейросеть
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_))'''

    # строим графики потерь и точности
    N = np.arange(0, epochs)
    #plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_accuracy")
    plt.plot(N, history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
#plt.savefig(args["plot"])