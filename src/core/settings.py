from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GroqConfig(BaseSettings):
    api_key: SecretStr
    base_url: str
    model: str
    timeout: float = 60.0

    model_config = SettingsConfigDict(
        env_prefix="groq_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class ChatSettings(BaseSettings):
    max_retries: int = 3
    retry_max_wait: int = 20
    log_dir: str = "logs"
    log_level: str = "INFO"

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


class ElevenlabsConfig(BaseSettings):
    api_key: SecretStr | None = None
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"

    model_config = SettingsConfigDict(
        env_prefix="elevenlabs_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None


class Settings(BaseSettings):
    groq: GroqConfig = Field(default_factory=lambda: GroqConfig())  # type: ignore[call-arg]
    chat: ChatSettings = Field(default_factory=lambda: ChatSettings())
    embedder: EmbedderConfig = Field(default_factory=lambda: EmbedderConfig())  # type: ignore[call-arg]
    elevenlabs: ElevenlabsConfig = Field(default_factory=lambda: ElevenlabsConfig())


settings = Settings()
