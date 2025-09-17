"""
Configuration management for Rocky AI
Centralized configuration with environment variable support
"""
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="rocky_ai", env="DB_NAME")
    user: str = Field(default="rocky", env="DB_USER")
    password: str = Field(default="secure_password", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class DMRConfig(BaseSettings):
    """Docker Model Runner configuration"""
    base_url: str = Field(default="http://localhost:11434", env="DMR_BASE_URL")
    timeout: int = Field(default=300, env="DMR_TIMEOUT")
    max_retries: int = Field(default=3, env="DMR_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="DMR_RETRY_DELAY")


class ExecutorConfig(BaseSettings):
    """Code executor configuration"""
    max_execution_time: int = Field(default=30, env="EXECUTOR_MAX_TIME")
    max_memory_mb: int = Field(default=512, env="EXECUTOR_MAX_MEMORY")
    max_output_size: int = Field(default=1048576, env="EXECUTOR_MAX_OUTPUT")  # 1MB
    working_dir: str = Field(default="/tmp/rocky_executors", env="EXECUTOR_WORK_DIR")
    cleanup_after: int = Field(default=3600, env="EXECUTOR_CLEANUP_AFTER")  # 1 hour


class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class CacheConfig(BaseSettings):
    """Cache configuration"""
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")  # 1 hour
    analysis_ttl: int = Field(default=7200, env="CACHE_ANALYSIS_TTL")  # 2 hours
    model_ttl: int = Field(default=86400, env="CACHE_MODEL_TTL")  # 24 hours
    max_memory: str = Field(default="256mb", env="CACHE_MAX_MEMORY")


class Settings(BaseSettings):
    """Main application settings"""
    app_name: str = Field(default="Rocky AI", env="APP_NAME")
    version: str = Field(default="0.2.0", env="APP_VERSION")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Service configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    dmr: DMRConfig = Field(default_factory=DMRConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Feature flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_database: bool = Field(default=True, env="ENABLE_DATABASE")
    enable_executors: bool = Field(default=True, env="ENABLE_EXECUTORS")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_database_url() -> str:
    """Get database URL"""
    return settings.database.url


def get_redis_url() -> str:
    """Get Redis URL"""
    return settings.redis.url


def is_development() -> bool:
    """Check if running in development mode"""
    return settings.environment == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production mode"""
    return settings.environment == Environment.PRODUCTION
