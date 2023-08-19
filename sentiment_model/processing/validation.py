from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(inputs=validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    Id:Optional[int]
    ProductId:Optional[str]
    UserId:Optional[str]
    ProfileName:Optional[str]
    HelpfulnessNumerator:Optional[int]
    HelpfulnessDenominator:Optional[int]
    Score:Optional[int]
    Time:Optional[str]
    Summary:Optional[str]
    Text:Optional[str]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]