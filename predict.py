import numpy as np

def calculate_predictions_log(tx, w):
    predictions = []
    for i in range(tx.shape[0]):
        pred = sigmoid(tx[i,].dot(w))
        predictions.append(int(pred > 0.5))
    return np.array(predictions)

def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred
