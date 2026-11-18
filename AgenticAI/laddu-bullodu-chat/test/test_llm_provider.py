import pytest
from langchain_core.tools import StructuredTool

import src.llm_provider as provider


@pytest.mark.parametrize(
    ("model_name", "key_name", "constructor_name", "expected_kwargs"),
    [
        (
            provider.GEMINI_MODEL,
            "GOOGLE_API_KEY",
            "ChatGoogleGenerativeAI",
            {"model": provider.GEMINI_MODEL, "google_api_key": "test-key"},
        ),
        (
            provider.OPENROUTER_FREE_MODEL,
            "OPENROUTER_API_KEY",
            "ChatOpenAI",
            {
                "model": provider.OPENROUTER_FREE_MODEL,
                "api_key": "test-key",
                "base_url": provider.OPENROUTER_BASE_URL,
            },
        ),
    ],
)
def test_new_provider_models_require_keys_and_use_expected_client(
    monkeypatch, model_name, key_name, constructor_name, expected_kwargs
):
    """Create the new provider clients with the configured model and API key."""
    monkeypatch.setenv(key_name, "test-key")
    provider.get_chat_model.cache_clear()
    calls = []

    def fake_constructor(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(provider, constructor_name, fake_constructor)
    try:
        assert provider.get_chat_model(model_name) is not None
        assert calls == [expected_kwargs | ({"default_headers": {"X-OpenRouter-Title": "Laddu Bullodu Chat"}} if model_name == provider.OPENROUTER_FREE_MODEL else {})]
    finally:
        provider.get_chat_model.cache_clear()


@pytest.mark.parametrize(
    ("model_name", "key_name"),
    [
        (provider.GEMINI_MODEL, "GOOGLE_API_KEY"),
        (provider.OPENROUTER_FREE_MODEL, "OPENROUTER_API_KEY"),
    ],
)
def test_new_provider_models_report_missing_credentials(monkeypatch, model_name, key_name):
    """Raise a provider-specific setup error when a new cloud key is absent."""
    monkeypatch.delenv(key_name, raising=False)
    if model_name == provider.GEMINI_MODEL:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider.get_chat_model.cache_clear()
    with pytest.raises(ValueError, match=key_name):
        provider.get_chat_model(model_name)
    provider.get_chat_model.cache_clear()


def test_groq_gpt_oss_model_uses_groq_key(monkeypatch):
    """Make Groq GPT-OSS selectable through the existing Groq API credential."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider.get_chat_model.cache_clear()
    calls = []
    monkeypatch.setattr(
        provider,
        "ChatGroq",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    try:
        provider.get_chat_model(provider.GROQ_GPT_OSS_MODEL)
        assert calls == [
            {"model": provider.GROQ_GPT_OSS_MODEL, "max_tokens": 380}
        ]
    finally:
        provider.get_chat_model.cache_clear()


def test_gemini_and_openrouter_clients_support_tool_binding(monkeypatch):
    """Verify both new client integrations expose the tool binding used by LangGraph."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    tool = StructuredTool.from_function(
        func=lambda query: query,
        name="delegate",
        description="Delegate a task.",
    )

    for model_name in (provider.GEMINI_MODEL, provider.OPENROUTER_FREE_MODEL):
        provider.get_chat_model.cache_clear()
        model = provider.get_chat_model(model_name)
        assert model.bind_tools([tool], tool_choice="any") is not None

    provider.get_chat_model.cache_clear()
