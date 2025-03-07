import tensorflow as tf
from tensorflow.keras import layers

class ConvLSTM2D(layers.Layer):
    def __init__(self, filters, kernel_size, activation='tanh', return_sequences=False, padding='same', **kwargs):
        super(ConvLSTM2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.return_sequences = return_sequences
        self.padding = padding
        
        # Convolutional layer (Conv2D)
        self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                  activation=self.activation, padding=self.padding)

        # LSTM layer (ConvLSTM)
        self.lstm = layers.LSTM(filters, return_sequences=self.return_sequences)

    def call(self, inputs):
        # Apply Conv2D to the input
        x = self.conv(inputs)

        # Apply LSTM on the output of Conv2D
        x = self.lstm(x)

        return x
