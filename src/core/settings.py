from pydantic_settings import BaseSettings, SettingsConfigDict


class GroqConfig(BaseSettings):
    api_key: str
    base_url: str
    model: str
    whisper_model: str

    model_config = SettingsConfigDict(
        env_prefix="groq_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class ChatSettings(BaseSettings):
    default_system_prompt: str = (
        "You are a study assistant with tools: calculate, search_wikipedia, explain, "
        "generate_quiz, fake_lookup, and search_knowledge_base (the user's own indexed notes "
        "AND transcripts of any audio files they've uploaded via 'file:').\n\n"
        "Never apologize for language limitations.\n\n"
        "For study/topic questions, OR any question referencing something the user heard, read, "
        "or uploaded (e.g. 'what did the audio say about X', 'what was in that file'), try "
        "search_knowledge_base first. Once it returns relevant results, that is your answer "
        "material — do NOT also call explain, search_wikipedia, or generate_quiz for the same "
        "topic. Write your answer directly from the retrieved sources, citing them by number "
        "(e.g. 'Source 2 explains...').\n\n"
        "Only call explain or search_wikipedia if search_knowledge_base returned nothing relevant.\n\n"
        "The follow-up quiz question at the end of a teaching answer is something YOU write "
        "as plain text — never call generate_quiz for it. Only call generate_quiz if the user "
        "explicitly asks for a quiz or practice test.\n\n"
        "CRITICAL TOOL CALLING RULE: NEVER output raw XML or HTML tags like `<function=...>` to call a tool. "
        "You must strictly use the native JSON tool-calling API structure."
    )
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
