import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

from ..config import settings


class ModelExporter:
    """Handle model export in different formats"""

    def __init__(self, model_path: str):
        """Initialize exporter with model path."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

    def export_joblib(self) -> str:
        """
        Export model as joblib pickle.

        Returns:
            Path to exported file
        """
        export_id = str(uuid.uuid4())[:8]
        export_filename = f"model_{export_id}.joblib"
        export_path = settings.MODELS_DIR / export_filename

        # Load and re-save to ensure compatibility
        model = joblib.load(self.model_path)
        joblib.dump(model, export_path)

        return str(export_path)

    def export_onnx(self) -> str:
        """
        Export model as ONNX.

        Returns:
            Path to exported file

        Note:
            Only works with scikit-learn models that have ONNX support.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnx
        except ImportError:
            raise RuntimeError("ONNX export requires: pip install skl2onnx onnx")

        # Load model
        model = joblib.load(self.model_path)

        # Get initial types (assumes float input)
        initial_type = [('float_input', FloatTensorType([None, None]))]

        try:
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            # Save ONNX model
            export_id = str(uuid.uuid4())[:8]
            export_filename = f"model_{export_id}.onnx"
            export_path = settings.MODELS_DIR / export_filename

            onnx.save(onnx_model, str(export_path))

            return str(export_path)

        except Exception as e:
            raise RuntimeError(f"Failed to convert model to ONNX: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the model."""
        try:
            model = joblib.load(self.model_path)

            info = {
                'type': type(model).__name__,
                'module': type(model).__module__,
                'has_predict': hasattr(model, 'predict'),
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'has_feature_importances': hasattr(model, 'feature_importances_'),
                'has_coef': hasattr(model, 'coef_'),
                'file_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'created_at': datetime.fromtimestamp(
                    self.model_path.stat().st_mtime
                ).isoformat()
            }

            # Try to get params
            if hasattr(model, 'get_params'):
                try:
                    info['params'] = str(model.get_params())
                except:
                    pass

            return info

        except Exception as e:
            return {'error': str(e)}
