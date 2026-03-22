"""
jd_analyzer.py
--------------
Uses **LangChain** with Google Gemini (LLM #1) to extract structured
requirements from a raw job description text.

LangChain serves as the orchestrator for:
  - Prompt templating (ChatPromptTemplate)
  - LLM invocation (ChatGoogleGenerativeAI via langchain-google-genai)
  - Structured JSON output parsing (JsonOutputParser)

The LLM parses the job description and returns a JSON object with:
  - title          : Job title
  - must_have      : List of mandatory skills / qualifications
  - nice_to_have   : List of optional / preferred qualifications
  - experience_min : Minimum years of experience (int)
  - keywords       : Domain-specific keywords to look for in CVs
  - summary        : 2-3 sentence summary of the role

This module is deliberately kept independent of cv_scorer.py so each
LLM step is easy to swap, test, or extend without touching the other.
"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import llm_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM #1 - gemini-2.5-flash: fast, efficient model used for structured
# extraction from the job description. This task is moderate complexity --
# pulling a fixed JSON schema out of a short document -- and Flash handles
# it accurately. Runs only ONCE per session so speed is a bonus, not a need.
JD_MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an expert HR consultant and technical recruiter. "
    "Your task is to analyse a job description and extract the key hiring "
    "criteria in structured JSON format. Be precise, concise, and realistic."
)

_USER_PROMPT = """
Analyse the following job description and extract the hiring criteria.

Return a JSON object with EXACTLY these keys:
  "title"          : string  - exact job title
  "must_have"      : list of strings - mandatory technical/non-technical requirements
  "nice_to_have"   : list of strings - preferred but not mandatory qualifications
  "experience_min" : integer - minimum years of relevant experience (0 if unspecified)
  "keywords"       : list of strings - important domain/tool/language keywords
  "summary"        : string - 2-3 sentence role summary

Output ONLY valid JSON. No markdown, no extra text.

Job Description:
---
{jd_text}
---
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_job_description(jd_text: str, model: str = JD_MODEL) -> dict:
    """
    Extract structured hiring criteria from a job description using LangChain.

    LangChain orchestrates the full chain:
      ChatPromptTemplate → ChatGoogleGenerativeAI → JsonOutputParser

    Parameters
    ----------
    jd_text : str
        Raw text of the job description.
    model : str
        Gemini model name to use. Defaults to JD_MODEL constant.
        Overridable via the --llm1 CLI argument in main.py.

    Returns
    -------
    dict
        Parsed JSON dict with keys: title, must_have, nice_to_have,
        experience_min, keywords, summary.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed as valid JSON.
    """
    print(f"[jd_analyzer] Analysing JD with LangChain + model: {model}")

    # Retrieve LangChain-wrapped Gemini LLM from shared client module
    llm = llm_client.get_langchain_llm(model=model, temperature=0.1)

    # Build LangChain chain: prompt → LLM → structured JSON parser
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_INSTRUCTION),
        ("human", _USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"jd_text": jd_text[:8000]})
        print(f"[jd_analyzer] LangChain chain completed successfully.")
        return result
    except Exception as exc:
        print(f"[jd_analyzer] LangChain chain failed, falling back to direct API: {exc}")
        return _fallback_direct_api(jd_text, model)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fallback_direct_api(jd_text: str, model: str) -> dict:
    """
    Fallback: call the Gemini API directly if the LangChain chain fails.
    """
    from google.genai import types

    client = llm_client.get()
    prompt = _USER_PROMPT.format(jd_text=jd_text[:8000])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

    raw_output = response.text.strip()
    return _parse_json_response(raw_output, label="JD analysis")


def _parse_json_response(raw: str, label: str = "response") -> dict:
    """
    Robustly parse a JSON object from an LLM response string.

    Strips markdown code fences if present before parsing.
    """
    # Strip markdown code fences  ```json ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try to extract the first {...} block as a fallback
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"[jd_analyzer] Failed to parse {label} as JSON.\n"
            f"Raw output:\n{raw}\n"
            f"Parse error: {exc}"
        )
