
import re
import sys
import typing as t
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import nltk
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def sentiment_imputer(dataframe: pd.Dataframe) -> pd.Dataframe:
    """
    Create a Sentiment column and assign values 1: if score >= 3, 0: if score < 3
    """

    df = dataframe.copy()
    df['Sentiment'] = np.where(df['Score'] >= 3, 1, 0)

    # drop duplicates if any w.r.t ['Text', 'Sentiment']
    df = df.drop_duplicates(subset = ['Sentiment', 'Text'],keep = 'last').reset_index(drop = True)

    return df


def remove_html_tags(input_text: str) -> str:
    """
    Remove HTML tags from a string
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', input_text)
   

def remove_punctuations(input_text: str) -> str:
    """
    Remove Punctuations
    """
    pattern = r'[^a-zA-Z0-9\s]'
    input_text = re.sub(pattern,'',input_text)
    # Single character removal
    input_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', input_text)
    # Removing multiple spaces
    input_text = re.sub(r'\s+', ' ', input_text)
    return input_text


def remove_stopwords(input_text: str, is_lower_case: bool) -> str:
    """
    Removing English Stop-words excluding negating words like 'not', 'won't', 'don't'...
    """

    updated_stopword_list = []
    # setting english stopwords
    stopword_list = nltk.corpus.stopwords.words('english') 

    # removing negating words from the standard list of stopwords
    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
            pass
        else:
            updated_stopword_list.append(word)

    # splitting strings into tokens (list of words)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in updated_stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in updated_stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def clean_text(dataframe: pd.Dataframe) -> pd.Dataframe:
    """
    Clean the 'Text' column:
        1. Remove HTML tags
        2. Remove punctuations
        3. Remove stopwords
    """
    df = dataframe.copy()
    df['Text'] = df['Text'].apply(remove_html_tags)
    df['Text'] = df['Text'].apply(remove_punctuations)
    df['Text'] = df['Text'].apply(remove_stopwords)

    return df


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # Create and impute Sentiment column
    data_frame = sentiment_imputer(dataframe=data_frame)

    # Clean the Text column
    data_frame = clean_text(dataframe=data_frame)

    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
