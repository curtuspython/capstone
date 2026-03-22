"""
jd_analyzer.py
--------------
Uses LLM #1 (Groq: llama-3.3-70b-versatile) to extract structured
requirements from a raw job description text.

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

from groq import Groq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM #1 – used for deep understanding of the job description
JD_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "You are an expert HR consultant and technical recruiter. "
    "Your task is to analyse a job description and extract the key hiring "
    "criteria in structured JSON format. Be precise, concise, and realistic."
)

_USER_PROMPT_TEMPLATE = """
Analyse the following job description and extract the hiring criteria.

Return a JSON object with EXACTLY these keys:
  "title"          : string  — exact job title
  "must_have"      : list of strings — mandatory technical/non-technical requirements
  "nice_to_have"   : list of strings — preferred but not mandatory qualifications
  "experience_min" : integer — minimum years of relevant experience (0 if unspecified)
  "keywords"       : list of strings — important domain/tool/language keywords
  "summary"        : string — 2-3 sentence role summary

Output ONLY valid JSON. No markdown, no extra text.

Job Description:
---
{jd_text}
---
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_job_description(jd_text: str, client: Groq) -> dict:
    """
    Extract structured hiring criteria from a job description.

    Parameters
    ----------
    jd_text : str
        Raw text of the job description.
    client : Groq
        Authenticated Groq API client.

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
    print(f"[jd_analyzer] Analysing JD with model: {JD_MODEL}")

    prompt = _USER_PROMPT_TEMPLATE.format(jd_text=jd_text[:8000])  # guard token limit

    response = client.chat.completions.create(
        model=JD_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,   # low temperature → consistent, deterministic output
        max_tokens=1024,
    )

    raw_output = response.choices[0].message.content.strip()
    return _parse_json_response(raw_output, label="JD analysis")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
