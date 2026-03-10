import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
from pathlib import Path

from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .exporter import ModelExporter
from ..utils.visualization import VisualizationService
from ..config import settings


class PipelineBuilder:
    """Orchestrate complete ML pipelines"""

    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.preprocessors = {}

    def run_pipeline(
        self,
        dataset_path: str,
        target_column: str,
        model_type: str,
        problem_type: str,
        preprocessing_config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete ML pipeline from data to model evaluation.

        Args:
            dataset_path: Path to dataset file
            target_column: Name of target column
            model_type: Type of model to train
            problem_type: 'classification', 'regression', or 'clustering'
            preprocessing_config: Preprocessing configuration
            hyperparameters: Model hyperparameters

        Returns:
            Dictionary with results and metadata
        """
        pipeline_id = str(uuid.uuid4())[:8]
        result = {
            'pipeline_id': pipeline_id,
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'model_id': None,
            'metrics': None,
            'charts': {},
            'error': None
        }

        try:
            # Load data
            df = DataLoader.load_dataset(dataset_path)
            is_valid, warnings = DataLoader.validate_dataset(df)

            if not is_valid:
                raise ValueError(f"Dataset validation failed: {warnings}")

            # Split data
            X_train, X_test, y_train, y_test = DataLoader.split_data(
                df, target_column, test_size=0.2, random_state=42
            )

            # Preprocess
            preprocessor = Preprocessor(preprocessing_config)
            X_train_processed = preprocessor.fit_transform(X_train, y_train)
            X_test_processed = preprocessor.transform(X_test)

            # Train
            trainer = ModelTrainer()
            trainer.train(
                X_train_processed,
                y_train,
                model_type,
                problem_type,
                hyperparameters
            )

            # Evaluate
            y_pred = trainer.predict(X_test_processed)
            feature_names = X_train_processed.columns.tolist()

            if problem_type == 'classification':
                y_proba = trainer.predict_proba(X_test_processed)
                metrics = ModelEvaluator.evaluate_classification(y_test, y_pred, y_proba)

                # Generate charts
                charts = VisualizationService.generate_classification_charts(
                    y_test.values, y_pred, y_proba,
                    trainer.get_feature_importance(feature_names)
                )

            elif problem_type == 'regression':
                metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)

                # Generate charts
                charts = VisualizationService.generate_regression_charts(
                    y_test.values, y_pred,
                    trainer.get_feature_importance(feature_names)
                )

            elif problem_type == 'clustering':
                metrics = ModelEvaluator.evaluate_clustering(X_test_processed, y_pred)
                charts = {}
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")

            # Save model and preprocessor
            model_id = str(uuid.uuid4())[:8]
            model_path = settings.MODELS_DIR / f"model_{model_id}.joblib"
            preprocessor_path = settings.MODELS_DIR / f"preprocessor_{model_id}.joblib"

            trainer.save(str(model_path))
            preprocessor.save(str(preprocessor_path))

            # Store references
            self.models[model_id] = {
                'path': str(model_path),
                'type': model_type,
                'problem_type': problem_type,
                'timestamp': datetime.now().isoformat()
            }
            self.preprocessors[model_id] = str(preprocessor_path)

            result.update({
                'status': 'completed',
                'model_id': model_id,
                'metrics': metrics,
                'charts': charts,
                'feature_importance': trainer.get_feature_importance(feature_names),
                'test_samples_count': len(X_test),
                'train_samples_count': len(X_train),
            })

        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e)
            })

        return result

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return self.models.get(model_id)

    def list_models(self) -> list:
        """List all trained models."""
        return list(self.models.values())

    def export_model(
        self,
        model_id: str,
        format: str = 'joblib'
    ) -> Dict[str, str]:
        """
        Export a trained model.

        Args:
            model_id: ID of model to export
            format: Export format ('joblib' or 'onnx')

        Returns:
            Export information
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model_info = self.models[model_id]
        exporter = ModelExporter(model_info['path'])

        if format == 'joblib':
            output_path = exporter.export_joblib()
        elif format == 'onnx':
            output_path = exporter.export_onnx()
        else:
            raise ValueError(f"Unknown export format: {format}")

        return {
            'model_id': model_id,
            'format': format,
            'path': output_path,
            'download_url': f"/api/models/{model_id}/download/{format}"
        }
