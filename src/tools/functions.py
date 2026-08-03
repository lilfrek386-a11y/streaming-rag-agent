import json
import math
import wikipedia


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        if isinstance(result, int) and abs(result).bit_length() > 1000:
            return json.dumps({"error": "Result too large to compute safely"})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_wikipedia(query: str, lang: str = "en") -> str:
    try:
        wikipedia.set_lang(lang)
        summary = wikipedia.summary(query, sentences=3, auto_suggest=False)
        return json.dumps({"query": query, "summary": summary}, ensure_ascii=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return json.dumps({"error": f"Query is ambiguous, try one of: {e.options[:5]}"}, ensure_ascii=False)
    except wikipedia.exceptions.PageError:
        return json.dumps({"error": f"No Wikipedia page found for '{query}' in language '{lang}'"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def explain(topic: str, lang: str = "en") -> str:
    try:
        wikipedia.set_lang(lang)
        summary = wikipedia.summary(topic, sentences=5, auto_suggest=False)
        return json.dumps({
            "topic": topic,
            "explanation": summary,
            "note": "Simplify this for a beginner, use an analogy or example if helpful."
        }, ensure_ascii=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return json.dumps({"error": f"Topic is ambiguous, try one of: {e.options[:5]}"}, ensure_ascii=False)
    except wikipedia.exceptions.PageError:
        return json.dumps({
            "topic": topic,
            "note": f"No reference article found for '{topic}' — explain it from your own knowledge, simply and with an example."
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_quiz(theme: str, difficulty: str = "medium") -> str:
    result = {
        "theme": theme,
        "difficulty": difficulty,
        "quiz_config": {
            "total_questions": 3,
            "format": "multiple_choice",
            "options_per_question": 4
        },
        "status": "ready_to_generate"
    }
    return json.dumps(result, ensure_ascii=False)


def fake_lookup(query: str) -> str:
    mock_db = {
        "python": "Python was created by Guido van Rossum and released in 1991.",
        "ai": "Artificial Intelligence reference data: Machine Learning models optimize parameters via gradient descent.",
        "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
        "photosynthesis": "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy.",
        "quantum computing": "Quantum computing leverages superposition and entanglement to solve complex mathematical problems.",
        "docker": "Docker delivers software in isolated packages called containers using OS-level virtualization.",
        "blackwell": "NVIDIA Blackwell is a GPU microarchitecture designed for large-scale generative AI and accelerated computing.",
        "database": "A database is an organized collection of structured information or data stored electronically in a computer system."
    }

    query_clean = query.lower().strip()

    result_data = None
    for key, value in mock_db.items():
        if key in query_clean or query_clean in key:
            result_data = value
            break

    if not result_data:
        result_data = f"Simulated Record: Detailed factual entry for '{query}' found in internal DB v2.4 (Confidence score: 0.98)."

    return json.dumps({
        "query": query,
        "source": "Internal_Encyclopedia_DB",
        "result": result_data
    }, ensure_ascii=False)


TOOL_FUNCTIONS = {
    "calculate": calculate,
    "search_wikipedia": search_wikipedia,
    "explain": explain,
    "generate_quiz": generate_quiz,
    "fake_lookup": fake_lookup,
}