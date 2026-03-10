from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import uuid

from ..models.schemas import (
    PipelineConfig, PipelineResult, PreprocessingConfig, ModelType, ProblemType
)
from ..services.pipeline_builder import PipelineBuilder
from ..routers.datasets import datasets_metadata

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])

# Global pipeline builder instance
pipeline_builder = PipelineBuilder()
pipelines_history = {}


@router.post("/run")
async def run_pipeline(config: Dict[str, Any]) -> PipelineResult:
    """
    Run a complete ML pipeline.

    Request body example:
    {
        "dataset_id": "abc123",
        "target_column": "target",
        "model_type": "random_forest",
        "problem_type": "classification",
        "preprocessing": {
            "scaling": {"type": "standard"},
            "encoding": {"type": "label", "columns": ["feature1"]}
        }
    }
    """
    try:
        # Validate dataset exists
        dataset_id = config.get('dataset_id')
        if dataset_id not in datasets_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset_path = datasets_metadata[dataset_id]['path']

        # Run pipeline
        result = pipeline_builder.run_pipeline(
            dataset_path=dataset_path,
            target_column=config.get('target_column'),
            model_type=config.get('model_type'),
            problem_type=config.get('problem_type'),
            preprocessing_config=config.get('preprocessing', {}),
            hyperparameters=config.get('hyperparameters')
        )

        # Store in history
        pipelines_history[result['pipeline_id']] = result

        return PipelineResult(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/")
async def list_pipelines() -> List[Dict[str, Any]]:
    """List all pipeline executions."""
    return list(pipelines_history.values())


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str) -> Dict[str, Any]:
    """Get pipeline execution details."""
    if pipeline_id not in pipelines_history:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return pipelines_history[pipeline_id]


@router.get("/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str) -> Dict[str, str]:
    """Get pipeline execution status."""
    if pipeline_id not in pipelines_history:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline = pipelines_history[pipeline_id]
    return {
        'pipeline_id': pipeline_id,
        'status': pipeline['status'],
        'timestamp': pipeline['timestamp'],
        'model_id': pipeline.get('model_id')
    }
