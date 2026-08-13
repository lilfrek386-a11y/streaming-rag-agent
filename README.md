# AI Agent CLI — Tool Calling Assistant with RAG

A powerful, multi-turn CLI assistant powered by LLM that uses tool/function calling to simulate educational tools, including semantic search over a personal knowledge base (RAG). This chatbot responds to study-related questions and autonomously delegates tasks like explanations, quiz generation, math calculations, or retrieval-augmented answers to custom Python functions.

Uses the OpenAI SDK pointed at Groq via `base_url`.

## How It Works (Under the Hood)

This assistant is built using an **Agentic Loop** with a **Hybrid Routing (Two-Pass)** architecture:

1. **Decision & Routing (Pass 1 - No Stream):** 
   When the user asks a question, the prompt and the registered JSON tool schemas are sent to the LLM with `stream=False`. This ensures reliable JSON parsing if the model decides to invoke a tool.
2. **Execution & Parallel Calling:** 
   If tools are called, the script detects `finish_reason == "tool_calls"`. It parses the arguments, runs the local Python functions (handling multiple tools in parallel if requested), and appends the results to the chat history.
3. **Agentic Loop:** 
   The model evaluates the new context. If more tools are needed, it calls them (up to a defined `MAX_ROUNDS` to prevent infinite loops — a warning is printed if the loop still wanted to call tools when the limit was hit).
4. **Final Response (Pass 2 - Stream):** 
   Once the model has all the necessary facts from the tools, a final API call is made with `stream=True`. The model generates a human-readable response smoothly streamed to the terminal for excellent UX.

## Retrieval-Augmented Generation (RAG)

On startup, the app indexes a local knowledge base from `.txt` files:

1. **Loading & Chunking:** `src/corpus/*.txt` files are read and split into ~1500-character overlapping chunks (`src/retrieval/chunker.py`, via `langchain_text_splitters`), tagged with their source filename.
2. **Embedding:** Each chunk is encoded into a normalized vector using a local `sentence-transformers` model (`src/retrieval/embedder.py`) — no external embedding API required.
3. **Indexing:** Vectors are stored in a FAISS `IndexFlatIP` index (cosine similarity on normalized vectors), wrapped in a reusable `VectorStore` class (`add_text`, `add_many`, `search`, `get_by_id`).
4. **Retrieval:** The `search_knowledge_base` tool embeds the user's query, retrieves the top-N most similar chunks, and prints them to the terminal before the model answers.
5. **Generation:** The LLM synthesizes an answer from the retrieved chunks and cites them by number (e.g. "According to Source 2...").

## Available Tools (Python Functions)

The model can autonomously decide to use the following registered tools:
* `calculate(expression)`: Safely evaluates mathematical expressions (e.g., `137 * (58 + 91) ** 2`).
* `search_wikipedia(query, lang)`: Searches Wikipedia for factual summaries in the user's language.
* `explain(topic, lang)`: Uses Wikipedia to explain a study topic in simple terms.
* `generate_quiz(theme, difficulty)`: Creates a structured 3-question quiz on a given topic (Bonus).
* `fake_lookup(query)`: Simulates access to an internal reference database/encyclopedia (Bonus).
* `search_knowledge_base(query)`: Searches the user's own indexed `.txt` corpus for relevant passages, used for study/topic questions before falling back to `explain`/`search_wikipedia` (RAG).

## Project Structure

This project keeps a single, modular `src/` layout shared across the internship's CLI-assistant tasks rather than separate task-numbered folders — tools, retrieval, and corpus logic each live in their own module (`src/tools/`, `src/retrieval/`, `src/corpus/`) and are wired together in `main.py` and `chat_session.py`.

## Setup

    uv sync
    cp .env.sample .env   # fill in GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, EMBEDDER_MODEL_NAME

Place any `.txt` documents you want indexed into `src/corpus/`.

## Usage

Run the main file. You can inject a custom personality using the `-prompt` argument:

    uv run main.py -prompt "You are a helpful math and science tutor."

*Type `quit` / `exit` / `q` to end the session.*

## Example Output (Parallel Tool Calling)

    You: Explain the term "apophenia" and calculate what 137 * (58 + 91) ** 2 equals
      Calling tool: explain({"lang":"en","topic":"apophenia"})
      Calling tool: calculate({"expression":"137 * (58 + 91) ** 2"})
    [Assistant is typing...]
    Assistant: Apophenia is the human tendency to see patterns, connections, or meaning in random or unrelated data...

    Mathematical calculation:
    137 * (58 + 91)^2 = 3,041,537

    [Tokens used: 2588 | Total so far: 2588]

## Example Output (RAG — Semantic Search)

    You: what is fastapi?
      Calling tool: search_knowledge_base({"query":"what is fastapi"})

    -> Top 3 Matches:
    [1] (fastapi.txt) "As part of that, I needed to investigate, test and use many alternatives..."
    [2] (fastapi.txt) "FastAPI is a modern, fast (high-performance), web framework for building APIs..."
    [3] (fastapi.txt) "Development¶ By the time I started creating FastAPI itself..."

    Assistant: FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints...

    What are some potential applications of FastAPI, and how might it be used in real-world projects?
    [Tokens used: 8494 | Total so far: 19962]