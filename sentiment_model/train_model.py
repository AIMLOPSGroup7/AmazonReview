import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sentiment_model.config.core import config
from sentiment_model.model import classifier
from sentiment_model.processing.data_manager import load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model


def run_training() -> None:
    
    """
    Train the model.
    """
    train_data = load_train_dataset()
    val_data = load_validation_dataset()
    test_data = load_test_dataset()

    # Model fitting
    classifier.fit(train_data,
                   epochs = config.model_config.epochs,
                   validation_data = val_data,
                   callbacks = callbacks_and_save_model(),
                   verbose = config.model_config.verbose)

    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(test_data)
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

    
if __name__ == "__main__":
    run_training()