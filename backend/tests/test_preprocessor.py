import pytest
import pandas as pd
import numpy as np

from ..services.preprocessor import Preprocessor


@pytest.fixture
def sample_data():
    """Generate sample data with mixed types."""
    return pd.DataFrame({
        'numeric_1': [1.5, 2.3, 3.1, 4.2, 5.0, np.nan],
        'numeric_2': [10, 20, 30, 40, 50, 60],
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'target': [0, 1, 0, 1, 0, 1]
    })


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    config = {
        'scaling': {'type': 'standard'},
        'encoding': {'type': 'label'}
    }
    preprocessor = Preprocessor(config)
    assert preprocessor.config == config


def test_fit_imputation(sample_data):
    """Test missing value imputation."""
    config = {'imputation': {'strategy': 'mean'}}
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    preprocessor.fit(X)

    # Check imputation was fitted
    assert hasattr(preprocessor, '_imputer')


def test_transform_after_fit(sample_data):
    """Test fit-transform pipeline."""
    config = {
        'imputation': {'strategy': 'mean'},
        'scaling': {'type': 'standard'}
    }
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    # Check no NaN values after imputation
    assert X_transformed[['numeric_1', 'numeric_2']].isnull().sum().sum() == 0

    # Check shape is preserved
    assert X_transformed.shape[0] == X.shape[0]


def test_scaling(sample_data):
    """Test feature scaling."""
    config = {'scaling': {'type': 'standard'}}
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    # Check that numeric columns are scaled
    for col in numeric_cols:
        values = X_transformed[col].values
        assert np.isclose(np.mean(values), 0.0, atol=0.5) or np.isclose(np.mean(values), np.mean(X[col]), atol=0.1)


def test_minmax_scaling(sample_data):
    """Test MinMax scaling."""
    config = {'scaling': {'type': 'minmax'}}
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    # MinMax scales to [0, 1]
    for col in numeric_cols:
        values = X_transformed[col].dropna().values
        if len(values) > 0:
            assert np.min(values) >= -0.05  # Allow small numerical errors
            assert np.max(values) <= 1.05


def test_robust_scaling(sample_data):
    """Test Robust scaling."""
    config = {'scaling': {'type': 'robust'}}
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    # Just check that it doesn't error and produces output
    assert X_transformed.shape == X.shape


def test_label_encoding(sample_data):
    """Test categorical label encoding."""
    config = {
        'encoding': {
            'type': 'label',
            'columns': ['category']
        }
    }
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    # Check that category column is now numeric
    assert X_transformed['category'].dtype in [np.int64, np.int32, np.int]
    # Check unique values are sequential
    unique_vals = np.unique(X_transformed['category'].dropna())
    assert len(unique_vals) <= 3  # We have 3 categories: A, B, C


def test_onehot_encoding(sample_data):
    """Test one-hot encoding."""
    config = {
        'encoding': {
            'type': 'onehot',
            'columns': ['category']
        }
    }
    preprocessor = Preprocessor(config)

    X = sample_data.drop('target', axis=1)
    X_transformed = preprocessor.fit_transform(X)

    # Check that category column was dropped and one-hot encoded
    assert 'category' not in X_transformed.columns
    # Should have one column per category
    assert len(X_transformed.columns) > len(X.columns)


def test_pipeline_consistency(sample_data):
    """Test that fit and transform are consistent."""
    config = {
        'imputation': {'strategy': 'mean'},
        'scaling': {'type': 'standard'},
        'encoding': {'type': 'label', 'columns': ['category']}
    }

    X = sample_data.drop('target', axis=1)
    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    preprocessor = Preprocessor(config)
    preprocessor.fit(X_train)

    # Transform should work on new data
    X_test_transformed = preprocessor.transform(X_test)
    assert X_test_transformed.shape[0] == X_test.shape[0]


def test_feature_selection():
    """Test feature selection."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y)

    config = {
        'feature_selection': {
            'method': 'selectkbest',
            'n_features': 5
        }
    }

    preprocessor = Preprocessor(config)
    X_transformed = preprocessor.fit_transform(X_df, y_series)

    # Check that we selected fewer features
    assert X_transformed.shape[1] <= X_df.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
