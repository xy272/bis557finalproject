import os, sys
import numpy as np
import tensorflow.python as tf

# Build U-Net model
def FCN(IMG_HEIGHT, IMG_WIDTH, n=512, nBLOCK0=64, nBLOCK1=128):

    IMAGE_ORDERING =  "channels_last" 
    img_input = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,1)) 

    # Encoder Block 1
    x = tf.keras.layers.Conv2D(nBLOCK0, (5, 5), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = tf.keras.layers.Conv2D(nBLOCK0, (5, 5), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    block1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    # Encoder Block 2
    x = tf.keras.layers.Conv2D(nBLOCK1, (5, 5), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(block1)
    x = tf.keras.layers.Conv2D(nBLOCK1, (5, 5), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    ## bottoleneck    
    o = (tf.keras.layers.Conv2D(n, (int(IMG_HEIGHT/4), int(IMG_WIDTH/4)), activation='relu' , padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
    o = (tf.keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same', name="bottleneck_2", data_format=IMAGE_ORDERING))(o)
    # Decoder Block
    # upsampling to bring the feature map size to be the same as the input image i.e., heatmap size

    output = tf.keras.layers.Conv2DTranspose(15, kernel_size=(4,4), strides=(4,4), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)
    model = tf.keras.Model(img_input, output)
    return model