"""Environment-backed project settings."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Runtime settings for local development."""

    model_config = SettingsConfigDict(env_prefix="HANDSOMENET_", extra="ignore")

    data_dir: Path = Field(default=Path("data/raw/freihand"))
    processed_dir: Path = Field(default=Path("data/processed"))
    artifacts_dir: Path = Field(default=Path("artifacts"))


def get_settings() -> Settings:
    """Return validated project settings."""

    return Settings()

