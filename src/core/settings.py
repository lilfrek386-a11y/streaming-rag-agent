from pydantic_settings import BaseSettings, SettingsConfigDict


class GroqConfig(BaseSettings):
    api_key: str
    base_url: str
    model: str

    model_config = SettingsConfigDict(
        env_prefix="groq_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class ChatSettings(BaseSettings):
    max_retries: int = 3
    retry_max_wait: int = 20
    log_dir: str = "logs"

    model_config = SettingsConfigDict(
        env_prefix="chat_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class EmbedderConfig(BaseSettings):
    model_name: str

    model_config = SettingsConfigDict(
        env_prefix="embedder_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(BaseSettings):
    groq: GroqConfig = GroqConfig()
    chat: ChatSettings = ChatSettings()
    embedder: EmbedderConfig = EmbedderConfig()


settings = Settings()
