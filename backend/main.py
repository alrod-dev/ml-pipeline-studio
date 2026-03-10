from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

from .config import settings
from .routers import datasets, pipelines, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="End-to-End ML Pipeline Builder API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(datasets.router)
app.include_router(pipelines.router)
app.include_router(models.router)


@app.get("/")
async def root():
    """API root endpoint with service information."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "datasets": "/api/datasets",
            "pipelines": "/api/pipelines",
            "models": "/api/models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.API_TITLE
    }


@app.get("/api/info")
async def api_info():
    """Get API configuration information."""
    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "debug": settings.DEBUG,
        "upload_dir": str(settings.UPLOAD_DIR),
        "models_dir": str(settings.MODELS_DIR),
        "max_dataset_size_mb": settings.MAX_DATASET_SIZE_MB,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Models directory: {settings.MODELS_DIR}")
    logger.info(f"CORS origins: {settings.ALLOWED_ORIGINS}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info(f"Shutting down {settings.API_TITLE}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info"
    )
