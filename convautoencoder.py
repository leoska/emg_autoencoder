# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:04:56 2019

@author: leoska
"""

from tensorflow.keras.layers import Input, LeakyReLU, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from subplots import plot_history, network_evaluation

class ConvAutoEncoder:
    def __init__(self, signal_len = 481, channels = 1):
        self.signal_len = signal_len
        self.channels = channels
        self.optimizer = None
        self.encoder = None
        self.decoder = None
        self.model = None
        self.history = None
        self.score = None
        
    def _encoder(self):
        feature_maps = [16, 1]
        kernel_size = [3]
        input_window = Input(shape=(self.signal_len, 1))
        
        # Первый слой свертки
        x = Conv1D(feature_maps[0], kernel_size[0], padding = "same")(input_window)
        x = LeakyReLU(alpha=0.2)(x) # 481 dim
        
        # Слой подвыборки
        x = MaxPooling1D(pool_size = (2), padding = "same")(x) # 241 dim
        
        # Второй слой свертки
        x = Conv1D(feature_maps[1], kernel_size, padding = "same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Второй слой подвыборки
        # "encoded" is the encoded representation of the input
        encoded = MaxPooling1D(pool_size = (2), padding = "same")(x) # 121 dim
        
        # this model maps an input to its encoded representation
        encoder = Model(inputs = input_window, outputs = encoded)
        
        encoder.summary()
        #encoder.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])
        
        self.encoder = encoder
        return encoder
    
    def _decoder(self):
        feature_maps = [16, 1]
        kernel_size = [3, 2]
        encoded_window = Input(shape=(120, 1))
        
        x = Conv1D(feature_maps[1], kernel_size[0], padding = "same")(encoded_window)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = UpSampling1D(2)(x)
        
        x = Conv1D(feature_maps[0], kernel_size[1], padding = "same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = UpSampling1D(2)(x)
        
        x = Conv1D(feature_maps[1], kernel_size[0], padding = "same")(x)
        
        # "decoded" is the lossy reconstruction of the input
        decoded = LeakyReLU(alpha=0.2)(x)
        
        # create the decoder model
        decoder = Model(inputs = encoded_window, outputs = decoded)
        
        decoder.summary()
        
        #decoder.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])
        
        self.decoder = decoder
        return decoder
    
    def autoencoder(self):
        # Encoder model
        ec = self._encoder()
        
        print("error here")
        
        # Decoder model
        dc = self._decoder()
        
        print("error here2")
        
        # this model maps an input to its reconstruction
        input_signal = Input(shape=(self.signal_len, 1))
        ec_out = ec(input_signal)
        dc_out = dc(ec_out)
        model = Model(inputs = input_signal, outputs = dc_out)
        
        model.summary()
        
        # Get optimizer
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        # mse - mean_squared_error
        # categorical_crossentropy
        model.compile(optimizer = optimizer, loss = 'mse', metrics = ["accuracy"])
        
        self.optimizer = optimizer
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
        
    # Не работает, нехватает правильно установленной библиотеки graphviz
    def plot_model(self):
        plot_model(self.model, to_file='simplemodel.png', show_shapes=True)

    def network_evaluation(self, epochs = 90, batch_size = 64):
        network_evaluation(self.history, epochs, batch_size)
        
    def savemodel(self):
        pass
    