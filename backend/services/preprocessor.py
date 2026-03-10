import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline as SklearnPipeline
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


class Preprocessor:
    """Handle data preprocessing operations"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Dictionary with preprocessing steps
        """
        self.config = config or {}
        self.preprocessor_pipeline = None
        self.feature_names = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.label_encoders = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'Preprocessor':
        """
        Fit preprocessor on training data.

        Args:
            X: Input features
            y: Target variable (for feature selection)

        Returns:
            Self for chaining
        """
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

        # Handle imputation
        if 'imputation' in self.config:
            X = self._fit_imputation(X)

        # Handle categorical encoding
        if 'encoding' in self.config:
            X = self._fit_encoding(X)

        # Handle scaling
        if 'scaling' in self.config:
            X = self._fit_scaling(X)

        # Handle feature selection
        if 'feature_selection' in self.config and y is not None:
            X = self._fit_feature_selection(X, y)

        self.feature_names = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Input features to transform

        Returns:
            Transformed DataFrame
        """
        X = X.copy()

        # Apply imputation
        if 'imputation' in self.config and hasattr(self, '_imputer'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = self._imputer.transform(X[numeric_cols])

        # Apply encoding
        if 'encoding' in self.config:
            X = self._transform_encoding(X)

        # Apply scaling
        if 'scaling' in self.config and hasattr(self, '_scaler'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = self._scaler.transform(X[numeric_cols])

        # Apply feature selection
        if 'feature_selection' in self.config and hasattr(self, '_feature_selector'):
            selected_cols = self._get_selected_features()
            X = X[selected_cols]

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _fit_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer on numeric columns."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        strategy = self.config.get('imputation', {}).get('strategy', 'mean')

        self._imputer = SimpleImputer(strategy=strategy)
        X[numeric_cols] = self._imputer.fit_transform(X[numeric_cols])

        return X

    def _fit_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit categorical encoders."""
        encoding_config = self.config.get('encoding', {})
        encoding_type = encoding_config.get('type', 'label')
        columns = encoding_config.get('columns', self.categorical_columns)

        if encoding_type == 'label':
            for col in columns:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
        elif encoding_type == 'onehot':
            self._onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self._onehot_encoder.fit_transform(X[columns])
            encoded_cols = self._onehot_encoder.get_feature_names_out(columns)
            X = X.drop(columns=columns)
            X = pd.concat([X, pd.DataFrame(encoded, columns=encoded_cols, index=X.index)], axis=1)

        return X

    def _transform_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoders during transform."""
        encoding_config = self.config.get('encoding', {})
        encoding_type = encoding_config.get('type', 'label')
        columns = encoding_config.get('columns', self.categorical_columns)

        if encoding_type == 'label':
            for col in columns:
                if col in X.columns and col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        elif encoding_type == 'onehot' and hasattr(self, '_onehot_encoder'):
            encoded = self._onehot_encoder.transform(X[columns])
            encoded_cols = self._onehot_encoder.get_feature_names_out(columns)
            X = X.drop(columns=columns)
            X = pd.concat([X, pd.DataFrame(encoded, columns=encoded_cols, index=X.index)], axis=1)

        return X

    def _fit_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on numeric columns."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scaling_type = self.config.get('scaling', {}).get('type', 'standard')

        if scaling_type == 'standard':
            self._scaler = StandardScaler()
        elif scaling_type == 'minmax':
            self._scaler = MinMaxScaler()
        elif scaling_type == 'robust':
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()

        if len(numeric_cols) > 0:
            X[numeric_cols] = self._scaler.fit_transform(X[numeric_cols])

        return X

    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit feature selector."""
        method = self.config.get('feature_selection', {}).get('method', 'selectkbest')
        n_features = self.config.get('feature_selection', {}).get('n_features', min(10, X.shape[1]))

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return X

        # Determine if classification or regression
        is_classification = len(np.unique(y)) < max(10, len(y) / 100)
        score_func = f_classif if is_classification else f_regression

        if method == 'selectkbest':
            self._feature_selector = SelectKBest(score_func=score_func, k=min(n_features, len(numeric_cols)))
            self._feature_selector.fit(X[numeric_cols], y)

        return X

    def _get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        if hasattr(self, '_feature_selector'):
            numeric_cols = self.numeric_columns
            selected_mask = self._feature_selector.get_support()
            selected = [col for col, selected in zip(numeric_cols, selected_mask) if selected]
            return selected + self.categorical_columns
        return self.feature_names

    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'Preprocessor':
        """Load preprocessor from disk."""
        return joblib.load(path)
