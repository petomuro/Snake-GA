import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime


class NeuralNetwork:
    def __init__(self, no_of_layers, no_of_neurons, lr):
        self.no_of_layers = no_of_layers
        self.no_of_neurons = no_of_neurons
        self.lr = lr

    def model(self):
        self.network = Sequential()
        self.network.add(Input(shape=(8,)))

        for _ in range(self.no_of_layers):
            self.network.add(Dense(
                self.no_of_neurons, activation='relu'))

        self.network.add(Dense(4, activation='softmax'))

        self.network.summary()

        opt = Adam(learning_rate=self.lr)
        self.network.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

        return self.network

    def save_weights(self):
        self.network.save('weights/model' +
                          str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.h5')

    def load_weights_(self):
        self.network.load_weights('weights/model20210526044312.h5')
