import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import joblib
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


class ModelTrainer:
    """Handle model training with various algorithms"""

    CLASSIFIERS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'svm': SVC,
        'logistic_regression': LogisticRegression,
    }

    REGRESSORS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'svm': SVR,
        'linear_regression': LinearRegression,
    }

    CLUSTERERS = {
        'kmeans': KMeans,
    }

    def __init__(self):
        self.model = None
        self.model_type = None
        self.hyperparameters = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        problem_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> 'ModelTrainer':
        """
        Train a model.

        Args:
            X_train: Training features
            y_train: Training labels/values
            model_type: Type of model to train
            problem_type: 'classification', 'regression', or 'clustering'
            hyperparameters: Model hyperparameters

        Returns:
            Self for chaining
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}

        # Default hyperparameters
        params = self._get_default_params(model_type, problem_type)
        params.update(self.hyperparameters)

        if problem_type == 'classification':
            model_class = self.CLASSIFIERS[model_type]
        elif problem_type == 'regression':
            model_class = self.REGRESSORS[model_type]
        elif problem_type == 'clustering':
            model_class = self.CLUSTERERS[model_type]
            # For clustering, we only fit X
            self.model = model_class(**params)
            self.model.fit(X_train)
            return self
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        self.model = model_class(**params)
        self.model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (if available)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance if available."""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()

        if importance is None:
            return {}

        # Normalize importance scores
        if len(importance) != len(feature_names):
            return {}

        importance = importance / np.sum(importance)
        return {
            name: float(score)
            for name, score in zip(feature_names, importance)
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str):
        """Load model from disk."""
        return joblib.load(path)

    @staticmethod
    def _get_default_params(model_type: str, problem_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model."""
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
            },
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42,
            },
            'linear_regression': {},
            'kmeans': {
                'n_clusters': 3,
                'random_state': 42,
                'n_init': 10,
            },
        }

        return defaults.get(model_type, {})
