"""Shared chat-model configuration for the Streamlit and LangGraph workflows."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

OLLAMA_BASE_URL = "http://192.168.0.169:11434"
OLLAMA_MODEL = "llama3.2"
GROQ_MODEL = "qwen/qwen3.8-27b"
OPENAI_MODEL = "gpt-4o-mini"

MODEL_LABELS = {
    OLLAMA_MODEL: "Llama 3.2 (Ollama)",
    GROQ_MODEL: "Qwen 3.8 27B (Groq)",
    OPENAI_MODEL: "GPT-4o Mini (OpenAI)",
}


def trim_chat_messages(messages: list, max_chars: int = 16000) -> list:
    """Prioritize system instructions and recent conversation within a character budget.

    System content is retained first and truncated when necessary. Remaining
    messages are considered newest-first; the newest message is kept even if it
    alone exceeds the remaining budget. Supports both LangChain messages and
    role/content dictionaries.
    """
    system_messages = []
    conversation_messages = []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", "")
        if role == "system":
            system_messages.append(message)
        else:
            conversation_messages.append(message)

    selected = []
    remaining = max_chars
    bounded_system_messages = []
    for message in system_messages:
        if isinstance(message, dict):
            content = str(message.get("content", ""))[:remaining]
            bounded_system_messages.append({**message, "content": content})
            remaining -= len(content)
        else:
            content = str(getattr(message, "content", ""))[:remaining]
            bounded_system_messages.append(message.copy(content=content))
            remaining -= len(content)
        if remaining <= 0:
            break

    used_chars = max_chars - remaining
    for message in reversed(conversation_messages):
        content = getattr(message, "content", "") if not isinstance(message, dict) else message.get("content", "")
        content_length = len(str(content))
        if used_chars + content_length > max_chars and selected:
            break
        selected.append(message)
        used_chars += content_length
    return bounded_system_messages + list(reversed(selected))


@lru_cache(maxsize=3)
def get_chat_model(model_name: str = OLLAMA_MODEL):
    """Return a cached Ollama, Groq, or OpenAI chat client by configured model name.

    Groq and OpenAI require their respective API-key environment variables.
    The LRU cache holds at most one client per supported model; unknown names
    and missing credentials raise ``ValueError``.
    """
    if model_name == OLLAMA_MODEL:
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
            mirostat=0,
            base_url=OLLAMA_BASE_URL,
        )

    if model_name == GROQ_MODEL:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file to use Groq."
            )
        return ChatGroq(model=GROQ_MODEL, max_tokens=380)

    if model_name == OPENAI_MODEL:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use OpenAI."
            )
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)

    raise ValueError(f"Unsupported chat model: {model_name}")
