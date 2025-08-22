import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

def test_model_type():
    """
    Check that train_model returns a RandomForestClassifier instance on a small dataset.
    """
    X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Expected a RandomForestClassifier instance"


def test_compute_model_metrics_perfect():
    """
    Check compute_model_metrics returns 1.0 for precision, recall, and F1 on perfect predictions.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, "Precision should be 1.0"
    assert recall == 1.0, "Recall should be 1.0"
    assert fbeta == 1.0, "F1 should be 1.0"


def test_inference_values():
    """
    Check that inference returns only 0 or 1 values.
    """
    X_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    X_test = np.array([[0, 1], [1, 0]])

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert set(preds).issubset({0, 1}), "Predictions should only be 0 or 1"

def test_model_has_predict():
    """
    Check that the trained model has a predict method.
    """
    X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)

    assert hasattr(model, 'predict'), 'Trained model does not have a predict method'