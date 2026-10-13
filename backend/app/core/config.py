from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    PROJECT_NAME: str = "WorkPilot AI"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "default_secret_key_change_in_production"
    DEBUG: bool = True

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str = "default_jwt_secret_key_change_in_production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Ollama Local LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # CORS origins allowed for frontend communication
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database connection string (SQLAlchemy 2.x asyncpg)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:adarsh%402003@localhost:5432/MyDb"

    # Jira Cloud Configuration
    JIRA_BASE_URL: str = ""
    JIRA_EMAIL: str = ""
    JIRA_API_TOKEN: str = ""
    JIRA_TIMEOUT_SECONDS: float = 15.0

    # GitHub API Configuration
    GITHUB_TOKEN: str = ""
    GITHUB_OWNER: str = ""
    GITHUB_REPO: str = ""
    GITHUB_API_BASE_URL: str = "https://api.github.com"
    GITHUB_TIMEOUT_SECONDS: float = 15.0

    # Slack Web API Configuration
    SLACK_BOT_TOKEN: str = ""
    SLACK_DEFAULT_CHANNEL: str = ""
    SLACK_API_BASE_URL: str = "https://slack.com/api"
    SLACK_TIMEOUT_SECONDS: float = 15.0


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
