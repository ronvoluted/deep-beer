import numpy as np
import pandas as pd
import category_encoders as ce
from category_encoders import wrapper # wrapper is not found in category_encoders without this import
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from src.models.pytorch import PytorchDataset

def target_encode(X, y, cols, smoothing=0.5):
    """
    Use multiclass target encoding to turn categorical column into nominal numeric values
    
    Parameters
    ----------
    X : dataframe
        Dataset containing columns to encode
    y : dataframe
        Target column to base encoding on
    cols : list
        Columns that should be target encoded
    smoothing : float
        Ration of smoothing to perform, higher means more regularisation towards prior agerage

    Returns
    -------
    X : dataframe
        Dataframe with column/s target encoded
    multiclass_encoder : preprocessing object
        Encoder for further transforming
    """
    target_encoder = ce.TargetEncoder(cols=cols, smoothing=smoothing)

    multiclass_encoder = ce.wrapper.PolynomialWrapper(target_encoder)
    X = multiclass_encoder.fit_transform(X, y)

    return X, multiclass_encoder

def ordinal_encode(y, columns, dtype='int32'):
    """
    Turn target column into numeric representation
    
    Parameters
    ----------
    y : dataframe
        Target column to base encoding on
    columns : list
        Column header for target
    smoothing : float
        Ration of smoothing to perform, higher means more regularisation towards prior agerage

    Returns
    -------
    y : dataframe
        Dataframe with column/s target encoded
    ordinal_encoder : preprocessing object
        Encoder for further transforming
    """
    ordinal_encoder = OrdinalEncoder(dtype=dtype)

    y = ordinal_encoder.fit_transform(y)
    y = pd.DataFrame(y, columns=columns)

    return y, ordinal_encoder

def min_max_scale(X, columns):
    """
    Contrain values to 0-1 scale
    
    Parameters
    ----------
    X : dataframe
        Dataset containing columns to scale
    columns : list
        Columns that should be scaled
    smoothing : float

    Returns
    -------
    X : dataframe
        Dataframe with column/s scaled
    min_max_scaler : preprocessing object
        Scaler for further transforming
    """

    min_max_scaler = MinMaxScaler()
    X[columns] = min_max_scaler.fit_transform(X[columns])
    return X, min_max_scaler

def format_input(data, target_encoder, min_max_scaler, numeric_columns):
    """
    Format input data into tensor shape compatible with model predictions
    
    Parameters
    ----------
    data : dict
        Dictionary of values input to be used for prediciton
    target_encoder : preprocessing object
        Target encoder to be used for transform
    target_encoder : preprocessing object
        Target encoder to be used for transform
    min_max_scaler : preprocessing object
        Scaler to be used for transform
    numeric_columns : list
        Columns to be scaled

    Returns
    -------
    data : PytorchDataset
        Dataset of tensors (with empty y tensor) ready for predictions in model
    """
    data = pd.DataFrame(data, index=[0])
    data = target_encoder.transform(data)
    data[numeric_columns] = min_max_scaler.transform(data[numeric_columns])
    data = PytorchDataset(data, [-1])
    return data