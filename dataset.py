# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:56:58 2019

@author: leoska
"""

import random
import numpy as np
from tensorflow.python.keras.utils import to_categorical

def create_dataset(filepath):
    sgn = []
    lbl = []
    path = filepath + "/{}_data.txt"
    for i in range(0,10):
        data = np.loadtxt(path.format(i+1), dtype=np.float64)
        for j in range(np.shape(data)[0]):
            sgn.append(data[j, :])
            lbl.append(i)
    
    c = list(zip(sgn, lbl))
    random.shuffle(c)
    sgn, lbl = zip(*c)

    sgn = np.asarray(sgn, dtype=np.float64)
    lbl = np.asarray(lbl, dtype=np.int64)

    train_signals = sgn[0:int(0.6*len(sgn))]
    train_labels = lbl[0:int(0.6*len(lbl))]
    val_signals = sgn[int(0.6*len(sgn)):int(0.8*len(sgn))]
    val_labels = lbl[int(0.6*len(lbl)):int(0.8*len(lbl))]
    test_signals = sgn[int(0.8*len(sgn)):]
    test_labels = lbl[int(0.8*len(lbl)):]

    nrows, ncols = train_signals.shape
    train_signals = train_signals.reshape(nrows, ncols, 1)
    nrows, ncols = val_signals.shape
    val_signals = val_signals.reshape(nrows, ncols, 1)
    nrows, ncols = test_signals.shape
    test_signals = test_signals.reshape(nrows, ncols, 1)
    
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    test_labels = to_categorical(test_labels)

    return train_signals, train_labels, val_signals, val_labels, test_signals, test_labels

if __name__ == "__main__":
    # For test module
    files_path = "data_emg/" #path to your directory with 10 data.txt files
    train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset(files_path)