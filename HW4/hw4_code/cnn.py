from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 64
        self.epochs = 5
        self.init_lr= 1e-3 #learning rate

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''
        model = Sequential()
        model.add(tf.keras.Input(shape=(32,32,3)))
        model.add(Conv2D(8,3,padding="same"))
        model.add(LeakyReLU(0.10))
        model.add(Conv2D(32,3,padding="same"))
        model.add(LeakyReLU(0.10))
        model.add(MaxPooling2D())
        model.add(Dropout(0.30))
        model.add(Conv2D(32,3,padding="same"))
        model.add(LeakyReLU(0.10))
        model.add(Conv2D(64,3,padding="same"))
        model.add(LeakyReLU(0.10))
        model.add(MaxPooling2D())
        model.add(Dropout(0.30))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.10))
        model.add(Dropout(0.50))
        model.add(Dense(128))
        model.add(LeakyReLU(0.10))
        model.add(Dropout(0.50))
        model.add(Dense(10))
        model.add(tf.keras.layers.Softmax())

        return model

    
    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = self.create_net()
        self.model.compile(loss='categorical_crossentropy',metrics='accuracy')
        return self.model
