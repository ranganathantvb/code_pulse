from pathlib import Path
from typing import Optional

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    git_base_url: AnyHttpUrl = Field(default="https://api.github.com")
    git_token: Optional[str] = None

    sonar_base_url: AnyHttpUrl = Field(default="https://sonarcloud.io/api")
    sonar_token: Optional[str] = None

    jira_base_url: AnyHttpUrl = Field(default="https://your-domain.atlassian.net")
    jira_user_email: Optional[str] = None
    jira_api_token: Optional[str] = None

    rag_embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    data_dir: Path = Field(default=Path(".data"))


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
