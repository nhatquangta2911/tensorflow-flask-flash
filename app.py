from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import os
import json

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 

from init_model import create_model

app = Flask(__name__)
CORS(app, resources={f"/*": {"origins": "*"}})

@app.route("/")
def hello():
    return jsonify({"Greeting": "Hi! Anything new?"})

@app.route("/api/<string:id>/")
def get_user(id):
    return "User ID: %s" % id

@app.route("/demo", methods=["POST"])
def demo():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid request."})
    else:
        return data

@app.route("/save", methods=["GET"])
def save():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    model = create_model()

    model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images, test_labels),
          callbacks = [cp_callback])
    model.save('model.h5')
    return str(model)

@app.route("/predict", methods=["GET"])
def predict():
    new_model = keras.models.load_model('model.h5')

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    # loss, acc = new_model.evaluate(test_images, test_labels)
    predict = new_model.predict(np.expand_dims(test_images[29], 0))
    probabilities = sorted(predict[0], reverse = True)

    probabilities = [ { "result": str(np.where(predict[0] == probability)[0][0]), "probability": str(probability * 100) } for probability in probabilities ]

    return jsonify(probabilities[:3])

if __name__ == "__main__":
    app.run(debug=True)