import pandas as pd
from joblib import load
from modules.pytorch import PytorchDataset
from modules.performance import get_prediction

ordinal_encoder = load('./artefacts/ordinal_encoder.joblib')
target_encoder = load('./artefacts/target_encoder.joblib')
min_max_scaler = load('./artefacts/min_max_scaler.joblib')

def format_input(data, target_encoder, min_max_scaler, numeric_columns=['review_appearance', 'review_aroma', 'review_palate', 'review_taste']):
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
    print('Type of data is:\n', type(data))
    data = pd.DataFrame(data, index=[0])
    data = target_encoder.transform(data)
    data[numeric_columns] = min_max_scaler.transform(data[numeric_columns])
    data = PytorchDataset(data, [-1])
    return data

def process_input(input, model):
    """
    Process inputs from API request and return preductions
    
    Parameters
    ----------
    input : pydantic Reviews
        Dict-like object containing inputs to be used for prediction
    model : PytorchMultiClass
        Model to run predictions on

    Returns
    -------
    input : dict
        input with beer_style prediction appended
    """
    if isinstance(input, dict) != True:
      input = input.dict()

    dataset = format_input(input, target_encoder, min_max_scaler)
    prediction = get_prediction(dataset, model, ordinal_encoder)

    input['beer_style'] = prediction

    return input