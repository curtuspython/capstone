"""
llm_client.py
-------------
Centralised Google Gemini client initialisation using the new `google-genai`
SDK (replaces the deprecated `google-generativeai` package).

Pattern:
  - main.py calls `llm_client.init(api_key)` once at startup.
  - jd_analyzer.py and cv_scorer.py call `llm_client.get()` to retrieve
    the shared client instance without needing it passed as a parameter.

This mirrors the old `genai.configure()` global approach but uses the
proper SDK-recommended Client object.
"""

from google import genai

# Module-level client instance; populated by init()
_client: genai.Client | None = None


def init(api_key: str) -> None:
    """
    Initialise the Gemini client with the given API key.

    Must be called once in main.py before any LLM calls are made.

    Parameters
    ----------
    api_key : str
        A valid Google Gemini API key (starts with 'AIza...').
    """
    global _client
    _client = genai.Client(api_key=api_key)


def get() -> genai.Client:
    """
    Return the initialised Gemini client.

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
