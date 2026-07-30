from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class GroqConfig(BaseSettings):
    api_key: str | None = None
    base_url: str
    model: str

    model_config = SettingsConfigDict(
        env_prefix="groq_",
        env_file=ENV_FILE_PATH,
        extra="ignore"
    )

class ChatSettings(BaseSettings):
    default_system_prompt: str = "You are a helpful tutor."
    max_retries: int = 3
    retry_max_wait: int = 20
    log_dir: str = "logs"

    model_config = SettingsConfigDict(
        env_prefix="chat_",
        env_file=ENV_FILE_PATH,
        extra="ignore"
    )

class Settings(BaseSettings):
    groq: GroqConfig = GroqConfig()
    chat: ChatSettings = ChatSettings()



settings = Settings()