calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the exact numerical result. Use this for any arithmetic, exponents, roots, or numeric computation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '7**12' or '(25 * 4) + 17'",
                }
            },
            "required": ["expression"],
        },
    },
}

wikipedia_tool = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for a neutral factual summary — definitions of people, events, places, or things. Use this for 'what/who is X' style questions. For 'explain X to me' or teaching a concept, use explain instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or term to look up, e.g. 'French Revolution' or 'Guido van Rossum'",
                },
                "lang": {
                    "type": "string",
                    "description": "The 2-letter language code, e.g., 'ru' for Russian, 'en' for English, 'de' for German. Choose based on the user's input language.",
                },
            },
            "required": ["query", "lang"],
        },
    },
}

explain_tool = {
    "type": "function",
    "function": {
        "name": "explain",
        "description": "Explain a study concept in simple, beginner-friendly terms with an example — for when the user wants to LEARN or UNDERSTAND something (e.g. 'explain overfitting', 'what is photosynthesis and how does it work'). Use search_wikipedia instead for neutral factual lookups of people, events, or things.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The concept to explain, e.g. 'overfitting' or 'photosynthesis'",
                },
                "lang": {
                    "type": "string",
                    "description": "The 2-letter language code matching the user's input language, e.g. 'ru' for Russian, 'en' for English.",
                },
            },
            "required": ["topic", "lang"],
        },
    },
}

quiz_tool = {
    "type": "function",
    "function": {
        "name": "generate_quiz",
        "description": "Create a 3-question quiz structure on a given topic and difficulty level.",
        "parameters": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "The topic of the quiz, e.g., 'photosynthesis', 'Python basics'.",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Difficulty level of the quiz.",
                },
            },
            "required": ["theme", "difficulty"],
        },
    },
}

fake_lookup_tool = {
    "type": "function",
    "function": {
        "name": "fake_lookup",
        "description": "Look up a term in the internal mock reference database (a small fixed set of demo entries, not Wikipedia). Use ONLY when explicitly asked to check the internal/company database — for general knowledge or factual questions, use search_wikipedia or explain instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or term to look up in the internal database, e.g. 'python' or 'docker'.",
                }
            },
            "required": ["query"],
        },
    },
}

knowledge_base_tool = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the user's personal indexed knowledge base FIRST for any study or topic question — it contains curated notes on specific subjects (e.g. machine learning, FastAPI, etc.). ALWAYS try this tool before explain or search_wikipedia when the user asks to learn about, explain, or discuss a topic — the knowledge base may have more specific, curated content than general knowledge. Only skip it for pure definitions of unrelated people/places/things, or if it returns no relevant results. Cite the source number in your answer, e.g. 'According to Source 2...'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language question or topic to search for in the knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
}

tools = [
    calculator_tool,
    fake_lookup_tool,
    wikipedia_tool,
    explain_tool,
    quiz_tool,
    knowledge_base_tool,
]
