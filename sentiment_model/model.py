import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow import keras

from sentiment_model.config.core import config
from sentiment_model.processing.features import data_augmentation


# Create a function that returns a model
def create_model(input_shape, optimizer, loss, metrics):

    EMBEDDING_DIM = 32


    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length=maxlen))
    model.add(LSTM(units=40,  dropout=0.2, recurrent_dropout=0.2))
    #add Dense(1) layer with activation='sigmoid'
    model.add(Dense(1, activation='sigmoid' ))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Create model
classifier = create_model(input_shape = config.model_config.input_shape, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric])
