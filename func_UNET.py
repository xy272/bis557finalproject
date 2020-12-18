# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
import os, sys
import numpy as np
import tensorflow.python as tf

from tensorflow.python.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.python.keras.layers.core import Lambda, RepeatVector, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from tensorflow.python.keras.layers.merge import concatenate, add

def conv2d_block(input_tensor, NFILTER, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = NFILTER, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)   
    # second layer
    x = Conv2D(filters = NFILTER, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    return x

def unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, DROPOUT=0.1, NFILTER=16, BATCH=True):
    input_img = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, NFILTER * 1, kernel_size = 3, batchnorm = BATCH)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(DROPOUT)(p1)
    
    c2 = conv2d_block(p1, NFILTER * 2, kernel_size = 3, batchnorm = BATCH)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(DROPOUT)(p2)
    
    c3 = conv2d_block(p2, NFILTER * 4, kernel_size = 3, batchnorm = BATCH)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(DROPOUT)(p3)
    
    c4 = conv2d_block(p3, NFILTER * 8, kernel_size = 3, batchnorm = BATCH)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(DROPOUT)(p4)
    
    c5 = conv2d_block(p4, NFILTER = NFILTER * 16, kernel_size = 3, batchnorm = BATCH)
    
    # Expansive Path
    u6 = Conv2DTranspose(NFILTER * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(DROPOUT)(u6)
    c6 = conv2d_block(u6, NFILTER * 8, kernel_size = 3, batchnorm = BATCH)
    
    u7 = Conv2DTranspose(NFILTER * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(DROPOUT)(u7)
    c7 = conv2d_block(u7, NFILTER * 4, kernel_size = 3, batchnorm = BATCH)
    
    u8 = Conv2DTranspose(NFILTER * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(DROPOUT)(u8)
    c8 = conv2d_block(u8, NFILTER * 2, kernel_size = 3, batchnorm = BATCH)
    
    u9 = Conv2DTranspose(NFILTER * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(DROPOUT)(u9)
    c9 = conv2d_block(u9, NFILTER * 1, kernel_size = 3, batchnorm = BATCH)
    
    outputs = Conv2D(15, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_img], outputs=[outputs])
    return model