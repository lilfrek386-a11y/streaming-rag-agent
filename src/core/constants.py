FALLBACK_ENCODING = "cl100k_base"
QUIT_COMMANDS = ("quit", "exit", "q")
MAX_ROUNDS = 5

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

SUMMARY_PROMPTS = {
    "summary": "Summarize the following transcript in 2-4 clear sentences.",
    "extract_keywords": "Extract 5-10 key terms or topics from this transcript, as a comma-separated list.",
    "generate_title": "Generate one short, descriptive title (under 10 words) for this transcript.",
    "qna": "Generate 3 question-and-answer pairs based on the key points in this transcript.",
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
