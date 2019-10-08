# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:03:37 2019

@author: leoska
"""

from sklearn import metrics
import numpy as np
import seaborn as sns
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
    plt.title("График ошибки на тренировочных данных")
    plt.xlabel("Эпоха #")
    plt.ylabel("Значение ошибки")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("График ошибки на проверочных данных")
    plt.show()
    
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["acc"])
    plt.title("Доля верных ответов на тренировочных данных")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_acc"])
    plt.title("Доля верных ответов на проверочных данных")
    plt.show()
    
# Функция выводит сравнение графиков
def plot_examples(stock_input, stock_decoded, test_samples = 158, step = 22, colors = ['r', 'b']):
    n = 10  
    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(list(np.arange(0, test_samples, step))):
        # display original
        ax = plt.subplot(4, n, i + 1)
        if i == 0:
            ax.set_ylabel("Вход", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx], colors[0])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Выход", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx], colors[1])
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
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(N, history.history["loss"], label="Ошибка на тренировочных данных")
    plt.plot(N, history.history["val_loss"], label="Ошибка на проверочных данных")
    plt.title("Оценка ошибки при обучении")
    plt.xlabel("Эпоха #")
    plt.ylabel("Значени ошибки")
    plt.legend()
    
    ax = plt.subplot(1, 2, 2)
    plt.plot(N, history.history["acc"], label="train_accuracy")
    plt.plot(N, history.history["val_acc"], label="val_accuracy")
    plt.title("Доля верных ответов при обучении")
    plt.xlabel("Эпоха #")
    plt.ylabel("Доля верных ответов")
    plt.legend()
    plt.show()
#plt.savefig(args["plot"])
    
def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                #xticklabels=LABELS,
                #yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()