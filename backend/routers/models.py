from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
import pandas as pd

from ..services.pipeline_builder import PipelineBuilder
from ..services.trainer import ModelTrainer
from ..models.schemas import ModelInfo, ExportRequest, ExportResponse

router = APIRouter(prefix="/api/models", tags=["models"])

# Global pipeline builder instance (shared with pipelines router)
pipeline_builder = PipelineBuilder()


@router.get("/")
async def list_models() -> List[Dict[str, Any]]:
    """List all trained models."""
    models = pipeline_builder.list_models()
    return models


@router.get("/{model_id}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """Get model information."""
    model_info = pipeline_builder.get_model(model_id)

    if model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return model_info


@router.post("/{model_id}/export")
async def export_model(model_id: str, request: ExportRequest) -> ExportResponse:
    """
    Export a trained model in specified format.

    Supported formats:
    - joblib: Pickle-based format (recommended)
    - onnx: Open Neural Network Exchange format
    """
    try:
        export_info = pipeline_builder.export_model(model_id, request.format)

        return ExportResponse(
            filename=export_info['path'].split('/')[-1],
            format=request.format,
            download_url=export_info['download_url']
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}/download/{format}")
async def download_model(model_id: str, format: str) -> FileResponse:
    """
    Download exported model.

    Formats:
    - joblib
    - onnx
    """
    try:
        export_info = pipeline_builder.export_model(model_id, format)
        file_path = export_info['path']

        return FileResponse(
            path=file_path,
            filename=f"model_{model_id}.{format}",
            media_type="application/octet-stream"
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/predict")
async def predict(model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions using a trained model.

    Request body example:
    {
        "features": {
            "feature1": 1.5,
            "feature2": "category_A",
            ...
        }
    }
    """
    try:
        model_info = pipeline_builder.get_model(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load model
        model = ModelTrainer.load(model_info['path'])

        # Load preprocessor
        if model_id in pipeline_builder.preprocessors:
            from ..services.preprocessor import Preprocessor
            preprocessor = Preprocessor.load(pipeline_builder.preprocessors[model_id])

            # Create DataFrame from input
            input_df = pd.DataFrame([data.get('features', {})])

            # Preprocess
            input_processed = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(input_processed)[0]

            # Get probabilities if available
            proba = None
            if hasattr(model.model, 'predict_proba'):
                proba = model.predict_proba(input_processed)[0].tolist()

            return {
                'model_id': model_id,
                'prediction': float(prediction) if isinstance(prediction, (int, float)) else str(prediction),
                'probabilities': proba,
                'feature_importance': model.get_feature_importance(input_processed.columns.tolist())
            }

        raise HTTPException(status_code=400, detail="Preprocessor not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
