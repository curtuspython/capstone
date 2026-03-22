"""
llm_client.py
-------------
Centralised Google Gemini client initialisation providing three access layers:

  1. google-genai SDK client (v1beta)  -- general LLM inference (generate_content).
  2. google-genai SDK client (v1)      -- dedicated embedding client.
     Embedding calls (embed_content with text-embedding-004) require the
     stable v1 API endpoint.  LlamaIndex imports google.generativeai which
     sets global state that causes embed_content to route to v1beta where
     text-embedding-004 returns 404.  A separate Client instance with
     http_options={'api_version': 'v1'} bypasses this problem.
  3. LangChain ChatGoogleGenerativeAI -- prompt orchestration, score
     aggregation, and structured output parsing.
  4. LlamaIndex GeminiEmbedding       -- semantic matching framework
     (project requirement, attempted first; falls back to Tier 2 client).

Pattern:
  - main.py calls ``llm_client.init(api_key)`` once at startup.
  - Other modules call the appropriate getter to retrieve a shared instance.
"""

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Module-level state; populated by init()
_client: genai.Client | None = None        # default SDK client (v1beta)
_embed_client: genai.Client | None = None  # v1 client dedicated to embed_content
_api_key: str | None = None
_langchain_llms: dict = {}
_llama_embed = None


def init(api_key: str) -> None:
    """
    Initialise all Gemini clients with the given API key.

    Creates two genai.Client instances:
      - Default client (v1beta): used for generate_content (LLM inference).
      - Embedding client (v1): used for embed_content (text-embedding-004).
        LlamaIndex imports google.generativeai at module level which sets
        global state that routes embed_content to v1beta where the model
        returns 404.  A dedicated Client with api_version='v1' bypasses it.

    Must be called once before any LLM calls are made.

    Parameters
    ----------
    api_key : str
        A valid Google Gemini API key (starts with 'AIza...').
    """
    global _client, _embed_client, _api_key
    _api_key = api_key
    # Default client -- used for chat/generation (v1beta endpoint is fine here)
    _client = genai.Client(api_key=api_key)
    # Dedicated embedding client -- force v1 to avoid LlamaIndex v1beta contamination
    try:
        _embed_client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )
        print("[llm_client] Embedding client (v1) initialised.")
    except Exception as exc:
        print(f"[llm_client] Embedding client (v1) init failed: {exc}; will use default client.")
        _embed_client = _client


def get() -> genai.Client:
    """
    Return the default google-genai SDK client (v1beta endpoint).

    Used for LLM generate_content calls.  Do NOT use this for embed_content
    -- use get_embed_client() instead to avoid v1beta 404 errors.

    Returns
    -------
    genai.Client

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


def get_embed_client() -> genai.Client:
    """
    Return the v1-pinned google-genai SDK client for embed_content calls.

    text-embedding-004 only exists on the stable v1 API endpoint.  LlamaIndex
    imports google.generativeai which can set global state routing embed_content
    to v1beta where the model returns 404.  This client was initialised with
    ``http_options={'api_version': 'v1'}`` to bypass that problem.

    Returns
    -------
    genai.Client
        v1-pinned client (falls back to default client if v1 init failed).
    """
    if _embed_client is None:
        raise RuntimeError(
            "[llm_client] Embed client not initialised. "
            "Call llm_client.init(api_key) in main.py first."
        )
    return _embed_client


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

    LlamaIndex is used as the semantic matching framework (project requirement).
    Note: the actual embedding API calls in ranker.py are routed through the
    google.genai SDK client (get()) because llama-index-embeddings-gemini
    currently targets the deprecated google.generativeai (v1beta) package,
    whose text-embedding-004 endpoint returns 404 on the v1beta API version.
    This method is retained for compatibility and future use once the
    llama-index-embeddings-gemini package migrates to google.genai.

    Returns
    -------
    GeminiEmbedding or None
    """
    global _llama_embed
    if _llama_embed is None:
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding  # noqa: PLC0415
            _llama_embed = GeminiEmbedding(
                model_name="models/text-embedding-004",
                api_key=get_api_key(),
            )
        except Exception as exc:
            print(f"[llm_client] LlamaIndex embed init skipped: {exc}")
    return _llama_embed
