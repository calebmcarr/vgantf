'''
Name         :  generator.py
Author       :  Caleb Carr
Description  :  TensorFlow implementation of the Generator Model in
                Generating Videos with Scene Dynamics from Columbia.
                Some liberty was taken with activation layers
License      :  GNU V3
Date         :  30MAY2020
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
import numpy as np

def foreground():
    # kernal size (4,4,4) = 4
    # stride = (2,2,2) = 2
    # use Conv3DTranspose for upsampling
    in_shape = (1,1,100,1)
    fg = tf.keras.Sequential()
    fg.add(layers.Dense(4*4*2,use_bias=False,input_shape=(100,)))
    fg.add(layers.BatchNormalization())
    #fg.add(activations.tanh())

    fg.add(layers.Reshape((4,4,2,1)))

    #firt layer uses a (2,4,4) convolution; creates (4x4x2) from 100 dim Noise with 512 channels
    fg.add(layers.Conv3DTranspose(512,(2,4,4),strides=1,use_bias=False,padding='same'))
    #fg.add(layers.BatchNormalization())
    #fg.add(layers.LeakyReLU())

    #outputs 8x8x4 with 256 channels
    fg.add(layers.Conv3DTranspose(256,4,strides=2,use_bias=False,padding='same'))

    #outputs 16x16x8 with 128 channels
    fg.add(layers.Conv3DTranspose(128,4,strides=2,use_bias=False,padding='same'))

    #outputs 32x32x16 with 64 channels
    fg.add(layers.Conv3DTranspose(128,4,strides=2,use_bias=False,padding='same'))

    #outputs forground: 64x64x32 with 3 channels
    fg.add(layers.Conv3DTranspose(3,4,strides=2,use_bias=False,padding='same',activation='tanh'))

    return fg

def fg_mask(fg):
    mask = tf.keras.models.clone_model(fg)
    mask.add(layers.Conv3DTranspose(1,4,strides=1,use_bias=False,padding='same',activation='sigmoid'))
    return mask

def background():
    in_shape = (1,1,100,1)
    bg = tf.keras.Sequential()
    bg.add(layers.Dense(4*4,use_bias=False,input_shape=(100,)))
    bg.add(layers.BatchNormalization())
    bg.add(layers.LeakyReLU())

    bg.add(layers.Reshape((4,4,1,1)))

    bg.add(layers.Conv3DTranspose(512,(2,4,4),strides=1,use_bias=False,padding='same'))

    #outputs 8x8x4 with 256 channels
    bg.add(layers.Conv3DTranspose(256,4,strides=(2,2,1),use_bias=False,padding='same'))

    #outputs 16x16x8 with 128 channels
    bg.add(layers.Conv3DTranspose(128,4,strides=(2,2,1),use_bias=False,padding='same'))

    #outputs 32x32x16 with 64 channels
    bg.add(layers.Conv3DTranspose(128,4,strides=(2,2,1),use_bias=False,padding='same'))

    #outputs forground: 64x64x32 with 3 channels
    bg.add(layers.Conv3DTranspose(3,4,strides=(2,2,1),use_bias=False,padding='same',activation='tanh'))

    return bg

def video(m,f,b):
    '''Computes two-stream arch. to get generated video'''
    p1 = (m*f)
    p2 = (1-m)*b
    video = p1+p2
    return video

'''
#test casing
fg_model = foreground()
bg_model = background()
mask = fg_mask(fg_model)
# keras automatically fills Batches but tf.random.normal does not
noise = tf.random.normal([1,100])
gen_fg_vid = fg_model(noise,training=False)
gen_mask = mask(noise,training=False)
gen_bg_vid = bg_model(noise,training=False)
vid = video(gen_mask,gen_fg_vid,gen_bg_vid)
print(vid)
'''
