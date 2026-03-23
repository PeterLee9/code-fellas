from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    google_api_key: str
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str
    tavily_api_key: str

    # Gemini model names
    extraction_model: str = "gemini-3.1-flash-lite-preview"
    agent_model: str = "gemini-3.1-flash-lite-preview"
    embedding_model: str = "gemini-embedding-2-preview"

    # Scraping config
    scrape_delay_seconds: float = 2.0

    # Validation thresholds
    confidence_threshold: float = 0.7

    model_config = {
        "env_file": [".env", "backend/.env"],
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
