import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from ..services.trainer import ModelTrainer


@pytest.fixture
def classification_data():
    """Generate classification test data."""
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    return X_df, y_series


@pytest.fixture
def regression_data():
    """Generate regression test data."""
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    return X_df, y_series


def test_random_forest_classifier(classification_data):
    """Test Random Forest classifier."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'random_forest', 'classification')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_gradient_boosting_classifier(classification_data):
    """Test Gradient Boosting classifier."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'gradient_boosting', 'classification')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)


def test_svm_classifier(classification_data):
    """Test SVM classifier."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'svm', 'classification')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)


def test_logistic_regression_classifier(classification_data):
    """Test Logistic Regression classifier."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'logistic_regression', 'classification')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)


def test_linear_regression(regression_data):
    """Test Linear Regression."""
    X, y = regression_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'linear_regression', 'regression')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_random_forest_regressor(regression_data):
    """Test Random Forest regressor."""
    X, y = regression_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'random_forest', 'regression')

    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)


def test_feature_importance(classification_data):
    """Test feature importance extraction."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'random_forest', 'classification')

    importance = trainer.get_feature_importance(X_train.columns.tolist())
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert all(isinstance(v, float) for v in importance.values())


def test_predict_proba(classification_data):
    """Test probability predictions."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'logistic_regression', 'classification')

    probas = trainer.predict_proba(X_test)
    assert probas is not None
    assert probas.shape[0] == len(X_test)
    assert probas.shape[1] == 2


def test_hyperparameter_override(classification_data):
    """Test custom hyperparameters."""
    X, y = classification_data
    X_train, y_train = X[:80], y[:80]

    hyperparams = {'n_estimators': 50, 'max_depth': 5}
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, 'random_forest', 'classification', hyperparams)

    # Check that the model was trained (can make predictions)
    predictions = trainer.predict(X_train)
    assert len(predictions) == len(X_train)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
