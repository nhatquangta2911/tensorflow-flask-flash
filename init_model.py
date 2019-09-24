from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf 
from tensorflow import keras
import numpy as np 

# from utils import create_dataset

# (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()

# train_dataset = create_dataset(x_train, y_train)
# val_dataset = create_dataset(x_val, y_val)

# def load_model():
#     # Build model
#     model = keras.Sequential([
#     keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
#     keras.layers.Dense(units=256, activation='relu'),
#     keras.layers.Dense(units=192, activation='relu'),
#     keras.layers.Dense(units=128, activation='relu'),
#     keras.layers.Dense(units=10, activation='softmax')
#     ])
#     # Train model
#    model.compile(optimizer='adam', 
#               loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(
#     train_dataset.repeat(), 
#     epochs=10, 
#     steps_per_epoch=500,
#     validation_data=val_dataset.repeat(), 
#     validation_steps=2
# )

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return model
    

