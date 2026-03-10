from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import uuid
from pathlib import Path
import pandas as pd

from ..services.data_loader import DataLoader
from ..models.schemas import DatasetMetadata, DatasetStats
from ..config import settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# In-memory storage of dataset metadata
datasets_metadata = {}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> DatasetMetadata:
    """
    Upload a dataset file.

    Supported formats: CSV, JSON, XLSX
    """
    try:
        # Generate unique ID
        dataset_id = str(uuid.uuid4())[:8]

        # Save file
        file_path = settings.UPLOAD_DIR / f"{dataset_id}_{file.filename}"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        # Load and validate
        df = DataLoader.load_dataset(str(file_path))
        is_valid, warnings = DataLoader.validate_dataset(df)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset validation failed: {warnings}"
            )

        # Store metadata
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        metadata = {
            'id': dataset_id,
            'name': file.filename.rsplit('.', 1)[0],
            'filename': file.filename,
            'path': str(file_path),
            'rows': len(df),
            'columns': len(df.columns),
            'created_at': pd.Timestamp.now().isoformat(),
            'file_size_mb': file_size_mb
        }

        datasets_metadata[dataset_id] = metadata

        return DatasetMetadata(**metadata)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/")
async def list_datasets() -> List[DatasetMetadata]:
    """List all uploaded datasets."""
    return [DatasetMetadata(**meta) for meta in datasets_metadata.values()]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str) -> DatasetMetadata:
    """Get dataset metadata."""
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetMetadata(**datasets_metadata[dataset_id])


@router.get("/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: str) -> DatasetStats:
    """Get statistical summary of dataset."""
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = datasets_metadata[dataset_id]['path']
    df = DataLoader.load_dataset(dataset_path)
    stats = DataLoader.get_dataset_stats(df)

    return DatasetStats(
        id=dataset_id,
        name=datasets_metadata[dataset_id]['name'],
        **stats
    )


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10) -> dict:
    """Preview first N rows of dataset."""
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = datasets_metadata[dataset_id]['path']
    df = DataLoader.load_dataset(dataset_path)

    return {
        'rows': df.head(limit).to_dict(orient='records'),
        'total_rows': len(df),
        'columns': df.columns.tolist()
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict:
    """Delete a dataset."""
    if dataset_id not in datasets_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(datasets_metadata[dataset_id]['path'])
    if file_path.exists():
        file_path.unlink()

    del datasets_metadata[dataset_id]

    return {'message': f'Dataset {dataset_id} deleted'}
