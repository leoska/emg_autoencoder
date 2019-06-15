# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:10:30 2019

@author: leoska
"""

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from subplots import plot_history, network_evaluation

class SimpleAutoEncoder:
    def __init__(self, encoding_dim = [32], signal_len = 481, channels = 1):
        self.encoding_dim = encoding_dim
        self.signal_len = signal_len
        self.channels = channels
        self.encoder = None
        self.decoder = None
        self.model = None
        self.history = None
        self.score = None
        
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
        
        # this model maps an input to its reconstruction
        input_signal = Input(shape=(self.signal_len,))
        ec_out = ec(input_signal)
        dc_out = dc(ec_out)
        model = Model(inputs = input_signal, outputs = dc_out)
        
        model.summary()
        
        # mse - mean_squared_error
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])
        
        self.model = model
        return model
    
    def fit(self, train_data, validation_data, batch_size = 64, epochs = 90, shuffle = True, tb_logs = 'tb_logs'):
        tensorboard = TensorBoard(log_dir = tb_logs, histogram_freq = 1, write_graph = True, write_images = True)
        
        history = self.model.fit(train_data, train_data, 
                #steps_per_epoch=10,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data = validation_data,
                callbacks=[tensorboard],
                verbose=1)
        
        self.history = history
        return history
        
    def evaluate(self, test_data):
        score = self.model.evaluate(x = test_data, y = test_data, verbose = 0)
        self.score = score
        return score
    
    def plot_history(self):
        plot_history(self.history)

    def network_evaluation(self, epochs = 90, batch_size = 64):
        network_evaluation(self.history, epochs, batch_size)
        
    def savemodel(self):
        pass