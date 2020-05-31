'''
Name         :  discriminator.py
Author       :  Caleb Carr
Description  :  TensorFlow implementation of the Discriminator Model in
                Generating Videos with Scene Dynamics from Columbia.
                Some liberty was taken with activation layers
License      :  GNU V3
Date         :  30MAY2020
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

def discr():
    '''turns a 64x64x32, 3 channel video into a 1x1x1, 2 channel
       probability of real or fake'''
    in_shape = (64,64,32,3)
    disc = tf.keras.Sequential()
    disc.add(layers.Dense(64*64*32,use_bias=False,input_shape=in_shape))
    disc.add(layers.BatchNormalization())
    disc.add(layers.LeakyReLU())

    #use 3D convolutions to down sample several times
    # based off the graphic at http://www.cs.columbia.edu/~vondrick/tinyvideo/discriminator.png
    disc.add(layers.Conv3D(64,4,strides=2,use_bias=False,padding='same'))

    disc.add(layers.Conv3D(128,4,strides=2,use_bias=False,padding='same'))

    disc.add(layers.Conv3D(256,4,strides=2,use_bias=False,padding='same'))

    disc.add(layers.Conv3D(512,4,strides=2,use_bias=False,padding='same'))

    disc.add(layers.Conv3D(2,4,strides=(4,4,2),use_bias=False,padding='same'))

    return disc
