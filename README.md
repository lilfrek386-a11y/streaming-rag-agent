# AI Agent CLI — Tool Calling Assistant with RAG & Audio Transcription

[![CI](https://github.com/lilfrek386-a11y/ai_internship/actions/workflows/ci.yml/badge.svg)](https://github.com/lilfrek386-a11y/ai_internship/actions/workflows/ci.yml)

A multi-turn CLI assistant powered by an LLM that uses tool/function calling,
semantic search over a personal knowledge base (RAG), and speech-to-text audio
transcription. The assistant can search an indexed knowledge base and
summarize the current chat session on demand, and automatically decides
when to call these tools while answering.

Uses the OpenAI SDK pointed at Groq via `base_url`.

## How It Works (Under the Hood)

This assistant is built using an **Agentic Loop** with a **True Single-Stream Architecture**:

1. **Streaming with Tool Detection:**
   Every turn sends the prompt and registered tool schemas with `stream=True` and `stream_options={"include_usage": True}`. If the model generates a normal text response, tokens stream to the console in real time immediately.
2. **Streaming Tool Calls Accumulation:**
   If the model decides to invoke tools, incoming streaming deltas (`delta.tool_calls`) are dynamically accumulated chunk-by-chunk (`name`, `arguments`, `id`). JSON argument parsing occurs safely once the stream finishes.
3. **Tool Execution:**
   Accumulated tool calls are executed sequentially. Results are formatted into tool messages (`role="tool"`) and appended to the context.
4. **Agentic Loop:**
   The model evaluates the new context with tool outputs. If further tools are required, it continues the loop (up to `MAX_ROUNDS` to guard against runaway loops); otherwise, it streams the final synthesized answer.

Unlike legacy two-pass implementations, this design never repeats requests or discards generated tokens, minimizing latency and token costs.

### Turn Atomicity & Token Accounting

- **Atomicity:** `ChatSession.send_message` records the history length before appending the user message and rolls back the conversation if an error occurs mid-turn. This prevents leaving an `assistant` message with unfulfilled `tool_calls` in the context.
- **Accurate Token Tracking:** Token usage is accumulated per round directly from provider stream chunks (`chunk.usage`).

## Retrieval-Augmented Generation (RAG) & Diversity Re-ranking

On startup, the app indexes a local knowledge base from `.txt` files:

1. **Loading & Chunking:** `.txt` files in a `corpus/` directory at the
   project root (if present) are read and split into overlapping chunks
   (`src/retrieval/chunker.py`, via `langchain_text_splitters`), tagged with
   source filenames. New facts and audio transcripts can be added later at
   runtime through in-chat commands.
2. **Embedding:** Each chunk is encoded into a normalized vector using a
   local `sentence-transformers` model (`src/retrieval/embedder.py`).
3. **Indexing & Diversity Search:** Vectors are stored in a FAISS
   `IndexFlatIP` index. The search method implements **diversity
   re-ranking** (`max_per_source` constraint) so a single large document
   cannot dominate the top-N results.
4. **Retrieval:** The `semantic_search` tool returns the most relevant
   chunks (with source and score) for the model to use when answering.

### Index Persistence & Starter Corpus

The index is cached on disk in `.cache/kb/` (`index.faiss` plus `entries.json` holding chunk texts and metadata fingerprint). On startup the cache is reused only if the embedder model, chunk settings, and corpus files (size and mtime) are unchanged; otherwise the corpus is re-embedded. Anything added at runtime via `/update_kb_text` or `/update_kb_voice` is saved on exit. Use `--rebuild-kb` to ignore the cache.

> [!NOTE]
> **Demo / Sample Corpus Only:** The text files provided in `corpus/` (e.g., `fastapi.txt`, `ml.txt`) are **strictly a sample / mock dataset** included for demonstration, testing, and debugging purposes. In real-world usage, you should replace the contents of `corpus/` with your own documents or rely entirely on dynamic in-chat knowledge addition via `/update_kb_text` and `/update_kb_voice`. If `corpus/` is empty or deleted, initial indexing is simply skipped.

## Audio Capabilities (Whisper STT & ElevenLabs TTS)

- **Audio Transcription (STT):** Audio files (`.mp3`, `.wav`, `.m4a`) are transcribed via Groq's Whisper API (`whisper-large-v3-turbo`) and their transcripts are chunked and indexed into the FAISS vector store with source metadata.
- **Speech Synthesis (TTS):** Passing `--voice` enables spoken assistant responses using ElevenLabs. It defaults to the free standard system voice `George` (`JBFqnCBsd6RMkjVDRZzb`), avoiding paid library voice restrictions.

## Available Tools (Function Calling)

* `semantic_search(query, top_n=3)`: Searches the indexed knowledge base (text notes + audio transcripts) and returns the most relevant chunks.
* `summarize_session()`: Returns the conversation history of the current session so the model can synthesize a clean summary.

## In-Chat Commands

Type a message starting with `/` to run a command instead of sending it to the model:

| Command | Description |
|---|---|
| `/exit` | End the session |
| `/help` | List available commands |
| `/update_kb_text [text]` | Add a manual text fact to the knowledge base |
| `/update_kb_voice [path;path;...]` | Transcribe audio file(s) and add them to the knowledge base |
| `/change_prompt [text or file path]` | Change the system prompt mid-session |
| `/search [query]` | Force a `semantic_search` call and answer from the result |
| `/summarize_session` | Force a `summarize_session` call and summarize the chat so far |
| `/save_session [filename]` | Save the current message history to a JSON file |
| `/load_session [filename]` | Load a message history from a JSON file |
| `/retry` | Re-send your last message |

## Project Structure

```text
├── corpus/               # Sample knowledge base documents (.txt)
├── src/
│   ├── audio/            # Whisper transcription service
│   ├── chat/             # ChatSession orchestrator, streaming parser & tool execution
│   ├── commands/         # /-command router
│   ├── core/             # API client, settings, enums, constants, logging, prompts
│   ├── retrieval/        # Chunker, embedder, and FAISS VectorStore
│   ├── schemas/          # Pydantic models (chat, audio, tools)
│   └── tools/            # Tool execution functions
├── tests/                # Complete test suite (pytest)
├── main.py               # Application entrypoint & CLI loop
└── pyproject.toml        # Dependencies and project metadata
```

## Setup

```bash
uv sync --group dev
cp .env.sample .env
```

Fill in `.env`:
* `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL` — required.
* `EMBEDDER_MODEL_NAME` — required (`sentence-transformers` model).
* `ELEVENLABS_API_KEY` — optional, needed for `--voice` (TTS).
* `ELEVENLABS_VOICE_ID` — optional, defaults to free system voice `George`.

## Usage

```bash
# Basic run
uv run main.py

# With TTS voice output and specific persona
uv run main.py --voice -persona teacher

# Quick persona shortcuts
uv run main.py --voice -funny
uv run main.py -formal

# Custom system prompt
uv run main.py --prompt "You are a helpful math and science tutor."

# Force rebuilding knowledge base index
uv run main.py --rebuild-kb
```

## Testing & Quality Assurance

Run the test suite, linter, and static type checker:

```bash
# Run 43 automated unit & integration tests
uv run pytest

# Lint and code format checks
uv run ruff check .

# Static type verification
uv run mypy src tests
```

### Continuous Integration (CI)

Every push and pull request triggers automated GitHub Actions workflow (`.github/workflows/ci.yml`) which runs:
1. **Linter & Formatting:** `ruff check .`
2. **Type Checking:** `mypy src tests`
3. **Automated Testing:** `pytest` on Python 3.13 with `uv`.

## Known Limitations

* Conversation history is not truncated automatically, so extremely long sessions may eventually hit context limits.
* Tool calls within a single round run sequentially to maintain predictable local state.
