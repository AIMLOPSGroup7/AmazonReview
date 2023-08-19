
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

from sentiment_model.config.core import config
from sentiment_model.model import classifier
from sentiment_model.processing.data_manager import load_dataset, callbacks_and_save_model


def run_training() -> None:
    
    """
    Train the model.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here for reproducibility
        random_state=config.model_config.random_state,
    )

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
    #X_val_tok = tokenizer.texts_to_sequences(X_val)


   # Find the vocabulary size and perform padding on both train and test set
    vocab_size = len(tokenizer.word_index) + 1 

    maxlen = 100

    X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen=maxlen, truncating='post')
    X_test_pad = pad_sequences(X_test_tok, padding='post', maxlen=maxlen, truncating='post')

    # Model fitting
    
    classifier.fit(X_train_pad, y_train, batch_size=128, 
                   epochs=3, 
                   verbose=1, 
                   validation_split=0.2)

    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(test_data)
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

    
if __name__ == "__main__":
    run_training()