# AI Agent CLI — Tool Calling Assistant with RAG & Audio Transcription

A powerful, multi-turn CLI assistant powered by LLM that uses tool/function calling to simulate educational tools, semantic search over a personal knowledge base (RAG), and speech-to-text audio processing. This chatbot responds to study-related questions and autonomously delegates tasks like explanations, quiz generation, math calculations, RAG-retrieval, or audio summarization to custom Python functions.

Uses the OpenAI SDK pointed at Groq via `base_url`.

## How It Works (Under the Hood)

This assistant is built using an **Agentic Loop** with a **Hybrid Routing (Two-Pass)** architecture:

1. **Decision & Routing (Pass 1 - No Stream):** 
   When the user asks a question, the prompt and registered JSON tool schemas are sent to the LLM with `stream=False` to ensure reliable JSON parsing for tool invocation.
2. **Execution & Parallel Calling:** 
   If tools are called, the script detects `finish_reason == "tool_calls"`, parses arguments, runs local Python functions (handling multiple tools in parallel if requested), and appends results to the chat history.
3. **Agentic Loop:** 
   The model evaluates the new context. If more tools are needed, it calls them (up to a defined `MAX_ROUNDS` to prevent infinite loops).
4. **Final Response (Pass 2 - Stream):** 
   Once all necessary facts are gathered, a final API call is made with `stream=True` and streamed smoothly to the terminal.

## Retrieval-Augmented Generation (RAG) & Diversity Re-ranking

On startup, the app indexes a local knowledge base from `.txt` and audio transcripts:

1. **Loading & Chunking:** `src/corpus/*.txt` files and uploaded audio transcripts are read and split into overlapping chunks (`src/retrieval/chunker.py`, via `langchain_text_splitters`), tagged with source filenames.
2. **Embedding:** Each chunk is encoded into a normalized vector using a local `sentence-transformers` model (`src/retrieval/embedder.py`).
3. **Indexing & Diversity Search:** Vectors are stored in a FAISS `IndexFlatIP` index. The search method implements **Diversity Re-ranking** (`max_per_source` constraint) to prevent large documents from dominating the top-N results and giving short transcripts a fair chance.
4. **Generation:** The LLM synthesizes an answer from retrieved chunks and cites them by source number.

## Audio Transcription & Summarization (Whisper Integration)

You can send local audio files directly into the CLI session using the `file:` prefix:
* **Transcription:** Uses Groq's Whisper API (`whisper-large-v3` or specified model) with timestamps segmentation.
* **Summarization Modes:** Controlled via the `-mode` flag (`summary`, `extract_keywords`, `generate_title`, `qna`).
* **Auto-Indexing:** Transcripts are automatically chunked and indexed into the shared FAISS vector store, making voice notes instantly searchable via `search_knowledge_base`.
* **Markdown Logging:** Audio transcripts and summaries are automatically timestamped and saved into the `logs/` directory.

## Available Tools (Python Functions)

* `calculate(expression)`: Safely evaluates mathematical expressions.
* `search_wikipedia(query, lang)`: Searches Wikipedia for factual summaries.
* `explain(topic, lang)`: Explains study topics in simple, beginner-friendly terms.
* `generate_quiz(theme, difficulty)`: Creates a structured 3-question quiz.
* `fake_lookup(query)`: Simulates an internal reference database.
* `search_knowledge_base(query)`: Searches the user's indexed `.txt` corpus and audio transcripts (RAG).

## Project Structure

A modular `src/` layout:
* `src/audio/`: Transcriber and summarizer modules.
* `src/corpus/`: Corpus loader and text files.
* `src/core/`: Constants, settings (`pydantic-settings`), and API client with retry logic (`tenacity`).
* `src/retrieval/`: Chunker, embedder, and FAISS `VectorStore`.
* `src/tools/`: Tool schemas and Python functions (including closure-based dependency injection for the vector store).

## Setup

    uv sync
    cp .env.sample .env   # fill in GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, EMBEDDER_MODEL_NAME, GROQ_WHISPER_MODEL

## Usage

Run the main file. You can inject a custom personality or choose a summarization mode:

    uv run main.py -prompt "You are a helpful math and science tutor." -mode extract_keywords

*   **Chat normally** with text input.
*   **Upload audio** by typing: `file: path/to/audio.mp3` (supports batch processing with semicolons: `file: a.mp3;b.wav`).
*   *Type `quit` / `exit` / `q` to end the session.*