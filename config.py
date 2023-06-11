import os


class Configuration(object):
    DEBUG = False
    SECRET_KEY = os.environ.get('APP_SECRET_KEY', '1234')


FIRE_IMG_DIR = './fit_data/fire'
NOT_FIRE_IMG_DIR = './fit_data/not_fire'
MODELS_DIR = './models/fire-detector-resnet50.hdf5'
IMG_SIZE = (224, 224)

