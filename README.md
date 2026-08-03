# AI Agent CLI — Tool Calling Assistant

A powerful, multi-turn CLI assistant powered by LLM that uses tool/function calling to simulate educational tools. This chatbot responds to study-related questions and autonomously delegates tasks like explanations, quiz generation, or math calculations to custom Python functions.

Uses the OpenAI SDK pointed at Groq via `base_url`.

## How It Works (Under the Hood)

This assistant is built using an **Agentic Loop** with a **Hybrid Routing (Two-Pass)** architecture:

1. **Decision & Routing (Pass 1 - No Stream):** 
   When the user asks a question, the prompt and the registered JSON tool schemas are sent to the LLM with `stream=False`. This ensures reliable JSON parsing if the model decides to invoke a tool.
2. **Execution & Parallel Calling:** 
   If tools are called, the script detects `finish_reason == "tool_calls"`. It parses the arguments, runs the local Python functions (handling multiple tools in parallel if requested), and appends the results to the chat history.
3. **Agentic Loop:** 
   The model evaluates the new context. If more tools are needed, it calls them (up to a defined `MAX_STEPS` to prevent infinite loops).
4. **Final Response (Pass 2 - Stream):** 
   Once the model has all the necessary facts from the tools, a final API call is made with `stream=True`. The model generates a human-readable response smoothly streamed to the terminal for excellent UX.

## Available Tools (Python Functions)

The model can autonomously decide to use the following registered tools:
* `calculate(expression)`: Safely evaluates mathematical expressions (e.g., `137 * (58 + 91) ** 2`).
* `search_wikipedia(query, lang)`: Searches Wikipedia for factual summaries in the user's language.
* `explain(topic)`: Uses Wikipedia to explain a study topic in simple terms.
* `generate_quiz(theme, difficulty)`: Creates a structured 3-question quiz on a given topic (Bonus).
* `fake_lookup(query)`: Simulates access to an internal reference database/encyclopedia (Bonus).

## Setup

    uv sync
    cp .env.sample .env   # fill in GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

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