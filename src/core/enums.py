from enum import StrEnum


class PersonaMode(StrEnum):
    FUNNY = "funny"
    FORMAL = "formal"
    TEACHER = "teacher"


class WhisperResponseFormat(StrEnum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"
