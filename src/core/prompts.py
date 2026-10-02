from src.core.enums import PersonaMode

PERSONA_PROMPTS: dict[PersonaMode, str] = {
    PersonaMode.FUNNY: "You are a very funny and sarcastic AI assistant.",
    PersonaMode.FORMAL: "You are a highly formal and polite AI assistant",
    PersonaMode.TEACHER: "You are a patient and helpful teacher.",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
