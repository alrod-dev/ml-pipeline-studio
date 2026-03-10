import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Optional, Tuple


class ModelEvaluator:
    """Evaluate model performance"""

    @staticmethod
    def evaluate_classification(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classification model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

        # ROC AUC (for binary classification only)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except:
                metrics['roc_auc'] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            # Convert to serializable format
            metrics['classification_report'] = {
                k: {str(k2): v2 for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in report.items()
            }
        except:
            metrics['classification_report'] = None

        return metrics

    @staticmethod
    def evaluate_regression(
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate regression model.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)

        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def evaluate_clustering(
        X: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate clustering model.

        Args:
            X: Input features
            labels: Cluster labels

        Returns:
            Dictionary with metrics
        """
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            return {
                'n_clusters': int(len(np.unique(labels))),
                'silhouette_score': float(silhouette_score(X, labels)),
                'davies_bouldin_score': float(davies_bouldin_score(X, labels)),
                'samples_per_cluster': {
                    int(cluster): int(count)
                    for cluster, count in zip(*np.unique(labels, return_counts=True))
                }
            }
        except Exception as e:
            return {
                'n_clusters': int(len(np.unique(labels))),
                'error': str(e)
            }

    @staticmethod
    def get_prediction_stats(
        y_pred: np.ndarray,
        problem_type: str
    ) -> Dict[str, Any]:
        """Get statistics about predictions."""
        if problem_type == 'classification':
            unique_classes, counts = np.unique(y_pred, return_counts=True)
            return {
                'unique_predictions': int(len(unique_classes)),
                'class_distribution': {
                    str(cls): int(count)
                    for cls, count in zip(unique_classes, counts)
                }
            }
        elif problem_type == 'regression':
            return {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred)),
            }
        else:
            return {}
