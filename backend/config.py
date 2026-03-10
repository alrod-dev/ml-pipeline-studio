from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration"""

    # API Settings
    API_TITLE: str = "ML Pipeline Studio API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # File Storage
    UPLOAD_DIR: Path = Path("./uploads")
    MODELS_DIR: Path = Path("./models")
    TEMP_DIR: Path = Path("./temp")

    # ML Settings
    MAX_DATASET_SIZE_MB: int = 100
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
