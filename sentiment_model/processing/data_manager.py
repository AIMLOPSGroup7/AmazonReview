import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config