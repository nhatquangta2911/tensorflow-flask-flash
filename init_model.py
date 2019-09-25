from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf 
import numpy as np 

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return model



