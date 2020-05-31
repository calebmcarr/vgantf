'''
Name         :  encoder.py
Author       :  Caleb Carr
Description  :  TensorFlow implementation of the Encoder in
                Generating Videos with Scene Dynamics from Columbia.
                Some liberty was taken with activation layers

                Very similar to generator.py but functions are changed for input size
License      :  GNU V3
Date         :  30MAY2020
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
import numpy as np

def encoder():
    '''turns a 64x64, 3 channel image to a 4x4, 512 channel image'''
    in_shape = (64,64,3)
    enc = tf.keras.Sequential()
    enc.add(layers.Dense(64*64,use_bias=False,input_shape=in_shape))
    enc.add(layers.BatchNormalization())

    # encode 64x64 down to 4x4
    enc.add(layers.Conv2D(64,4,strides=2,use_bias=False,padding='same'))

    enc.add(layers.Conv2D(128,4,strides=2,use_bias=False,padding='same'))

    enc.add(layers.Conv2D(256,4,strides=2,use_bias=False,padding='same'))

    enc.add(layers.Conv2D(512,4,strides=2,use_bias=False,padding='same'))

    #this result will then be sent off to a modified Generator
    return enc

def foreground():
    # kernal size (4,4,4) = 4
    # stride = (2,2,2) = 2
    # use Conv3DTranspose for upsampling
    in_shape = (4,4,1)
    fg = tf.keras.Sequential()
    fg.add(layers.Dense(4*4,use_bias=False,input_shape=in_shape))
    fg.add(layers.BatchNormalization())
    #fg.add(activations.tanh())

    fg.add(layers.Reshape((4,4,1,1)))

    #firt layer uses a (2,4,4) convolution; creates (4x4x2) from 100 dim Noise with 512 channels
    fg.add(layers.Conv3DTranspose(512,(2,4,4),strides=(1,1,2),use_bias=False,padding='same'))


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
    in_shape = (4,4,1)
    bg = tf.keras.Sequential()
    bg.add(layers.Dense(4*4,use_bias=False,input_shape=in_shape))
    bg.add(layers.BatchNormalization())
    #fg.add(activations.tanh())

    bg.add(layers.Reshape((4,4,1,1)))

    #firt layer uses a (2,4,4) convolution; creates (4x4x2) from 100 dim Noise with 512 channels
    bg.add(layers.Conv3DTranspose(512,(2,4,4),strides=(1,1,2),use_bias=False,padding='same'))


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
#---TEST CASING---
#generate models
enc = encoder()
fg_model = foreground()
bg_model = background()
mask = fg_mask(fg_model)

#create noise tensor
noise = tf.random.normal([64,64])
#get encoded tensor from noise
gen_encoding = enc(noise,training=False)
#use this encoding to great normal generated video
gen_fg_vid = fg_model(gen_encoding,training=False)
gen_mask = mask(gen_encoding,training=False)
gen_bg_vid = bg_model(gen_encoding=False)
vid = video(gen_mask,gen_fg_vid,gen_bg_vid)
'''
