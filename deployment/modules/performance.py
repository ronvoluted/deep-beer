def decode_beer_style(encoded_target, encoder):
    """
    Get text representation of encoded target value
    
    Parameters
    ----------
    encoded_target : int32
        Value of target
    encoder : preprocessing object
        Encoder used for original encoding

    Returns
    -------
    original target value : str
        Decoded target value
    """
    return encoder.inverse_transform([[encoded_target]])[0][0]

def get_prediction(dataset, model, ordinal_encoder):
    """
    Generate prediction on input data
    
    Parameters
    ----------
    dataset : PytorchDataset
        Dataset containing tensor of input values
    model : PytorchMultiClass
        Trained nearal network
    ordinal_encoder : preprocessing object
        Ordinal encoder used to transform target column

    Returns
    -------
    predicted_style : str
        Predicted beer style
    """
    model.double()
    model.eval()

    predictions = model(dataset.X_tensor)

    class_probabilities = predictions[0].tolist()
    highest_prediction = max(class_probabilities)
    predicted = class_probabilities.index(highest_prediction)
    predicted_style = decode_beer_style(predicted, ordinal_encoder)

    return predicted_style
