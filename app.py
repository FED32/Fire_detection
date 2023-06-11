import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50
from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import BadRequestKeyError
import base64

import config
from config import Configuration
import logger

logger = logger.init_logger()

app = Flask(__name__)
app.config.from_object(Configuration)


class HttpError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message


def build_model():
    """загружает модель и передает в нее веса"""

    base_model = resnet50.ResNet50(weights='imagenet',
                                   include_top=False,
                                   input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    # x = tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l1(1e-4)
    )(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x, name='fire_detector')

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss)
                  metrics=['accuracy'])

    model.load_weights(config.MODELS_DIR)

    return model


work_model = build_model()


def load_image(img_string,
               target_size=config.IMG_SIZE):
    """функция для загрузки и предобработки изображения"""

    nparr = np.frombuffer(base64.b64decode(img_string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]
    img = cv2.resize(img, target_size)
    return resnet50.preprocess_input(img, data_format=None)


@app.route('/predict', methods=['POST'])
def predict():
    """метод для синхронной обработки, возвращает 1 - пожар, 0 - нет пожара"""

    try:
        json_file = request.get_json(force=False)
        img_string = json_file.get('img_string', None)
        prob = request.args.get('probability', 0)

        # data = load_image(img_string)
        data = load_image(img_string.encode('utf8'))

        result = work_model.predict(np.array([load_image(img_string)]), steps=1, verbose=1)
        logger.info("predict: success")

        if int(prob) == 1:
            return jsonify({'result': float(result[0][0])})

        return jsonify({'result': int(np.around(result)[0][0])})

    except BadRequestKeyError:
        logger.error("predict: BadRequest")
        return Response(None, 400)

    except KeyError:
        logger.error("predict: KeyError")
        return Response(None, 400)

    except BaseException as ex:
        logger.error(f'predict: {ex}')
        raise HttpError(400, f'{ex}')

