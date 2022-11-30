from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers


def data_preprocessing(IMG_SIZE=32):
    '''
    In this function you are going to build data preprocessing layers using tf.keras
    First, resize your image to consistent shape
    Second, standardize pixel values to [0,1]
    return tf.keras.Sequential object containing the above mentioned preprocessing layers
    '''
    # HINT :You can resize your images with tf.keras.layers.Resizing,
    # You can rescale pixel values with tf.keras.layers.Rescaling
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Resizing(IMG_SIZE,IMG_SIZE))
    model.add(tf.keras.layers.Rescaling(1./255))
    return model
    

def data_augmentation():
    '''
    In this function you are going to build data augmentation layers using tf.keras
    First, add random horizontal and vertical flip
    Second, add random rotation
    return tf.keras.Sequential object containing the above mentioned augmentation layers
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RandomFlip())
    model.add(tf.keras.layers.RandomRotation(0.5))
    #model.add(tf.keras.layers.RandomCrop())
    return model


    

