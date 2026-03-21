"""Application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for local development."""

    app_name: str = "AutoVisionLab API"
    app_version: str = "0.1.0"
    database_url: str = "sqlite:///./autovisionlab_demo.db"
    redis_url: str = "redis://localhost:6379/0"
    artifact_root: str = "artifacts"
    data_root: str = "data"
    is_demo_mode: bool = True
    demo_train_samples: int = 2000
    demo_val_samples: int = 1000
    aihubmix_api_key: str | None = None
    aihubmix_model: str = "minimax/minimax-m2.5"
    aihubmix_base_url: str = "https://aihubmix.com/v1"

    model_config = SettingsConfigDict(env_prefix="AVL_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cache settings so the app uses a single resolved instance."""
    return Settings()
