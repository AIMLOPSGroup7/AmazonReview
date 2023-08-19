import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from sentiment_model.config.core import config
from sentiment_model.train import vocab_size

# Create a function that returns a model
def create_model(input_shape, optimizer, loss, metrics):

    EMBEDDING_DIM = 32
    
    #tokenizer = Tokenizer(num_words=5000)
    maxlen=100
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length=maxlen))
    model.add(LSTM(units=40,  dropout=0.2, recurrent_dropout=0.2))
    #add Dense(1) layer with activation='sigmoid'
    model.add(Dense(1, activation='sigmoid' ))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


# Create model
classifier = create_model(input_shape = config.model_config.input_shape, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric])