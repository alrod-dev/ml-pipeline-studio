import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
from datetime import datetime
import os


class DataLoader:
    """Handle dataset loading and validation"""

    SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx']
    MAX_SIZE_MB = 100

    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """
        Load dataset from file with format detection.

        Args:
            file_path: Path to the dataset file

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If file format is not supported or loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > DataLoader.MAX_SIZE_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.2f}MB > {DataLoader.MAX_SIZE_MB}MB"
            )

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.xlsx':
            return pd.read_excel(file_path)
        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Supported: {DataLoader.SUPPORTED_FORMATS}"
            )

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataset integrity.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check if empty
        if df.empty:
            warnings.append("Dataset is empty")
            return False, warnings

        # Check for all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            warnings.append(f"All-null columns: {null_cols}")

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows")

        # Check for object columns with mostly missing values
        for col in df.select_dtypes(include='object').columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values"
                )

        return len(warnings) == 0, warnings

    @staticmethod
    def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistical summary of dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing statistical information
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {}
        }

        # Numeric statistics
        for col in numeric_cols:
            stats['numeric_stats'][col] = {
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'median': float(df[col].median()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None,
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'q25': float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                'q75': float(df[col].quantile(0.75)) if not df[col].isnull().all() else None,
            }

        # Categorical statistics
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            stats['categorical_stats'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': {str(k): int(v) for k, v in value_counts.items()}
            }

        return stats

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into train and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test
