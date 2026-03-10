import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from typing import Dict, Any, Optional, List
import json


class ChartGenerator:
    """Generate charts as base64-encoded images"""

    @staticmethod
    def confusion_matrix_chart(cm: np.ndarray, labels: Optional[List[str]] = None) -> str:
        """Generate confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> str:
        """Generate ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def feature_importance_chart(importance: Dict[str, float], top_n: int = 15) -> str:
        """Generate feature importance bar chart."""
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_importance = sorted_importance[:top_n]

        features = [item[0] for item in sorted_importance]
        scores = [item[1] for item in sorted_importance]

        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in scores]
        plt.barh(features, scores, color=colors)
        plt.xlabel('Importance Score')
        plt.title('Top Feature Importance')
        plt.tight_layout()

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def learning_curve_chart(
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = 'Score'
    ) -> str:
        """Generate learning curve."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)

        plt.plot(epochs, train_scores, 'b-', label='Training', marker='o')
        plt.plot(epochs, val_scores, 'r-', label='Validation', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(alpha=0.3)

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def distribution_chart(data: pd.Series, title: str = 'Distribution') -> str:
        """Generate distribution histogram."""
        plt.figure(figsize=(10, 6))
        plt.hist(data.dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(alpha=0.3, axis='y')

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 10)) -> str:
        """Generate correlation heatmap for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) == 0:
            return ""

        plt.figure(figsize=figsize)
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def scatter_plot(
        x_data: np.ndarray,
        y_data: np.ndarray,
        xlabel: str = 'X',
        ylabel: str = 'Y',
        title: str = 'Scatter Plot'
    ) -> str:
        """Generate scatter plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.6, s=50)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)

        return ChartGenerator._fig_to_base64()

    @staticmethod
    def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate residuals plot for regression."""
        residuals = y_true - y_pred

        plt.figure(figsize=(12, 5))

        # Residuals vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(alpha=0.3)

        # Residuals distribution
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return ChartGenerator._fig_to_base64()

    @staticmethod
    def _fig_to_base64() -> str:
        """Convert current matplotlib figure to base64."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"


class VisualizationService:
    """Service for generating visualizations for models and data"""

    @staticmethod
    def generate_classification_charts(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """Generate all charts for classification model."""
        charts = {}

        # Confusion matrix
        cm = pd.crosstab(y_true, y_pred)
        charts['confusion_matrix'] = ChartGenerator.confusion_matrix_chart(cm.values)

        # ROC curve (binary classification only)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            charts['roc_curve'] = ChartGenerator.roc_curve_chart(fpr, tpr, roc_auc)

        # Feature importance
        if feature_importance:
            charts['feature_importance'] = ChartGenerator.feature_importance_chart(feature_importance)

        return charts

    @staticmethod
    def generate_regression_charts(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """Generate all charts for regression model."""
        charts = {}

        # Residuals plot
        charts['residuals'] = ChartGenerator.residuals_plot(y_true, y_pred)

        # Predicted vs Actual
        charts['predicted_vs_actual'] = ChartGenerator.scatter_plot(
            y_true, y_pred,
            xlabel='Actual Values',
            ylabel='Predicted Values',
            title='Predicted vs Actual Values'
        )

        # Feature importance
        if feature_importance:
            charts['feature_importance'] = ChartGenerator.feature_importance_chart(feature_importance)

        return charts

    @staticmethod
    def generate_data_exploration_charts(df: pd.DataFrame) -> Dict[str, str]:
        """Generate exploratory charts for dataset."""
        charts = {}

        # Correlation heatmap
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            charts['correlation'] = ChartGenerator.correlation_heatmap(df)

        # Distributions of numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5
            charts[f'distribution_{col}'] = ChartGenerator.distribution_chart(
                df[col], title=f'Distribution of {col}'
            )

        return charts
