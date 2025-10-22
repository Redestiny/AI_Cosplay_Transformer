"""
Application settings and configuration
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    app_name: str = "InstantID Cosplay API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Storage Settings
    storage_type: str = "local"  # local, s3, gcs
    storage_path: str = "./storage"
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: str = "us-east-1"
    
    # Model Settings
    model_cache_dir: str = "./models"
    device: str = "cuda"  # cuda, cpu, mps
    max_concurrent_tasks: int = 4
    
    # Processing Settings
    default_resolution: str = "1024x1024"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list = ["jpg", "jpeg", "png", "webp"]
    
    # Security Settings
    allowed_origins: list = ["*"]
    max_requests_per_minute: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()
