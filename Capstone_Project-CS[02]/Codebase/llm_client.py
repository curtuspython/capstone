"""
llm_client.py
-------------
Centralised Google Gemini client initialisation providing three access layers:

  1. google-genai SDK client  – direct Gemini API (used where fine-grained
     control is needed, e.g. thinking_budget).
  2. LangChain ChatGoogleGenerativeAI – orchestrator for prompt-based
     matching, score aggregation, and structured output parsing.
  3. LlamaIndex Gemini embeddings – semantic candidate-job matching via
     vector similarity.

Pattern:
  - main.py / app.py calls ``llm_client.init(api_key)`` once at startup.
  - Other modules call the appropriate getter to retrieve a shared instance.
"""

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Module-level state; populated by init()
_client: genai.Client | None = None
_api_key: str | None = None
_langchain_llms: dict = {}
_llama_embed = None


def init(api_key: str) -> None:
    """
    Initialise all Gemini clients with the given API key.

    Must be called once before any LLM calls are made.

    Parameters
    ----------
    api_key : str
        A valid Google Gemini API key (starts with 'AIza...').
    """
    global _client, _api_key
    _api_key = api_key
    _client = genai.Client(api_key=api_key)


def get() -> genai.Client:
    """
    Return the raw google-genai SDK client.

    Returns
    -------
    genai.Client
        The shared client instance.

    Raises
    ------
    RuntimeError
        If called before llm_client.init() has been invoked.
    """
    if _client is None:
        raise RuntimeError(
            "[llm_client] Client not initialised. "
            "Call llm_client.init(api_key) in main.py first."
        )
    return _client


def get_api_key() -> str:
    """Return the stored API key."""
    if _api_key is None:
        raise RuntimeError(
            "[llm_client] API key not set. "
            "Call llm_client.init(api_key) first."
        )
    return _api_key


def get_langchain_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> "ChatGoogleGenerativeAI":
    """
    Return a LangChain ChatGoogleGenerativeAI instance (cached per config).

    LangChain is used as the orchestrator for prompt-based matching,
    score aggregation, and structured output parsing as required by
    the project specification.

    Parameters
    ----------
    model : str
        Gemini model name.
    temperature : float
        Sampling temperature.

    Returns
    -------
    ChatGoogleGenerativeAI
        Cached LangChain LLM instance.
    """
    # Lazy import so missing package gives a clear error only when used,
    # not at app startup before requirements are installed.
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

    cache_key = f"{model}_{temperature}"
    if cache_key not in _langchain_llms:
        _langchain_llms[cache_key] = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_api_key(),
            temperature=temperature,
        )
    return _langchain_llms[cache_key]


def get_llama_embed():
    """
    Return a LlamaIndex GeminiEmbedding instance for semantic matching.

    LlamaIndex is used as the framework for semantic candidate-job matching
    via vector similarity as required by the project specification.

    Returns
    -------
    GeminiEmbedding
    """
    global _llama_embed
    if _llama_embed is None:
        from llama_index.embeddings.gemini import GeminiEmbedding
        _llama_embed = GeminiEmbedding(
            model_name="models/text-embedding-004",
            api_key=get_api_key(),
        )
    return _llama_embed
