# Multi-Turn CLI Chat
 
Tutor-style CLI chat with streaming, token tracking, custom system prompts, and logging. Uses the OpenAI SDK pointed at Groq via `base_url`.
 
## Setup
 
```bash
  uv sync
  cp .env.sample .env   # fill in GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
  uv run main.py
```
 
## Usage
 
```bash
  uv run main.py -prompt "You are a math tutor."
```
 
Type `quit` / `exit` / `q` to end the session.