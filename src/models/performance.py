def print_class_perf(y_preds, y_actuals, set_name=None, average='binary'):
    """Print the Accuracy and F1 score for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    average : str
        Parameter  for F1-score averaging
    Returns
    -------
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    print(f"Accuracy {set_name}: {accuracy_score(y_actuals, y_preds)}")
    print(f"F1 {set_name}: {f1_score(y_actuals, y_preds, average=average)}")

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

def check_predictions(dataset, model, ordinal_encoder, num_checks=100):
    """
    Make a range of predictions and print actual results, showing which were correct and how many
    
    Parameters
    ----------
    dataset : PytorchDataset
        Dataset containing tensor of input values
    model : PytorchMultiClass
        Trained nearal network
    ordinal_encoder : preprocessing object
        Ordinal encoder used to transform target column
    num_checks : int32
        Number of predictions to check

    """
    model.double()
    model.eval()
    model.to('cpu')
    
    predictions = model(dataset.X_tensor)
    targets = dataset.y_tensor

    matches = 0
    
    for i in range(num_checks):
        class_probs = predictions[i].tolist()
        highest_prediction = max(class_probs)
        predicted = class_probs.index(highest_prediction)
        predicted_style = decode_beer_style(predicted, ordinal_encoder)
        
        actual = targets[i].item()
        actual_style = decode_beer_style(actual, ordinal_encoder)

        correct = False

        if predicted_style == actual_style:
            matches += 1
            correct = True

        print(f'{i}: {predicted_style} {"✅" if correct else f"| {actual_style}"}')

    print(f'{matches} out of {num_checks} correct')