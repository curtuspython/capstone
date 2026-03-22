"""
cv_scorer.py
------------
Uses LLM #2 (Google Gemini: gemini-2.5-flash) to score a single candidate
CV against the structured job requirements extracted by jd_analyzer.py (LLM #1).

For each CV the LLM returns:
  - overall_score      : int  0-100
  - must_have_score    : int  0-100  (match against mandatory requirements)
  - nice_to_have_score : int  0-100  (match against preferred qualifications)
  - experience_score   : int  0-100  (relevant experience fit)
  - keyword_score      : int  0-100  (keyword presence in CV)
  - strengths          : list[str] (top 3 candidate strengths)
  - gaps               : list[str] (top 3 missing areas)
  - recommendation     : str  (1-2 sentence recruiter note)

LLM #1 (gemini-2.5-pro)   -- deep semantic JD analysis, runs once per session.
LLM #2 (gemini-2.5-flash) -- fast per-CV scoring, runs once per candidate.
Using a heavier model for the complex one-time analysis and a lighter, faster
model for the repeated scoring loop is a standard two-LLM pipeline pattern.
"""

import json
import re

from google.genai import types

import llm_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM #2 - gemini-2.5-flash: a lighter, faster model used in the per-CV
# scoring loop. Thinking disabled for fast, deterministic JSON output.
# Deliberately different from LLM #1 (gemini-2.5-pro): the Pro model handles
# the complex one-time JD analysis; Flash handles the repeated scoring loop.
SCORER_MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are a senior technical recruiter. "
    "Evaluate candidate resumes against structured job requirements and "
    "provide objective, numeric scores with brief justifications."
)

_USER_PROMPT_TEMPLATE = """
Score the following candidate resume against the job requirements.

Job Requirements (JSON):
{requirements_json}

Candidate CV:
---
{cv_text}
---

Return a JSON object with EXACTLY these keys:
  "overall_score"      : integer 0-100
  "must_have_score"    : integer 0-100  (how well the candidate meets mandatory requirements)
  "nice_to_have_score" : integer 0-100  (how well the candidate meets preferred requirements)
  "experience_score"   : integer 0-100  (experience relevance and level)
  "keyword_score"      : integer 0-100  (presence of important domain keywords)
  "strengths"          : list of up to 3 short strings (what the candidate does well)
  "gaps"               : list of up to 3 short strings (what the candidate is missing)
  "recommendation"     : string (1-2 sentence recruiter note about this candidate)

Scoring rubric:
  90-100 : Exceptional match
  75-89  : Strong match, minor gaps
  60-74  : Good match, some training needed
  40-59  : Partial match, significant gaps
  0-39   : Poor match

Output ONLY valid JSON. No markdown, no extra text.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_cv(candidate: dict, requirements: dict, model: str = SCORER_MODEL) -> dict:
    """
    Score a single candidate CV against the extracted job requirements.

    Uses the shared Gemini client initialised by llm_client.init() in main.py.

    Parameters
    ----------
    candidate : dict
        Must contain keys 'name', 'file', and 'text' (raw CV text).
    requirements : dict
        Structured JD requirements from jd_analyzer.analyze_job_description().
    model : str
        Gemini model name to use. Defaults to SCORER_MODEL constant.
        Overridable via the --llm2 CLI argument in main.py.

    Returns
    -------
    dict
        The input candidate dict enriched with a 'scores' key containing
        the full scoring breakdown from the LLM.
    """
    candidate_name = candidate.get("name", "Unknown")
    print(f"[cv_scorer] Scoring: {candidate_name} (model: {model})")

    client = llm_client.get()  # retrieve shared client from llm_client module

    # Truncate CV text to avoid exceeding context length
    cv_text_trunc = candidate["text"][:6000]
    requirements_json = json.dumps(requirements, indent=2)

    prompt = _USER_PROMPT_TEMPLATE.format(
        requirements_json=requirements_json,
        cv_text=cv_text_trunc,
    )

    # Call the new google-genai SDK: system_instruction lives in GenerateContentConfig.
    # max_output_tokens raised to 4096 to ensure full JSON scoring output is returned.
    # thinking_config budget=0 disables chain-of-thought for fast, deterministic JSON.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.15,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw_output = response.text.strip()
    scores = _parse_json_response(raw_output, label=f"scores for {candidate_name}")

    # Return a new dict so original candidate is not mutated
    return {**candidate, "scores": scores}


def score_all_cvs(candidates: list[dict], requirements: dict, model: str = SCORER_MODEL) -> list[dict]:
    """
    Score every candidate in the list and return the enriched list.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dicts from resume_parser.parse_resumes_from_directory().
    requirements : dict
        Structured JD requirements from jd_analyzer.analyze_job_description().
    model : str
        Gemini model name to use for scoring. Overridable via --llm2 CLI arg.

    Returns
    -------
    list[dict]
        Each dict now includes a 'scores' key with numeric scores and narrative.
    """
    scored: list[dict] = []
    for candidate in candidates:
        try:
            scored.append(score_cv(candidate, requirements, model=model))
        except Exception as exc:
            print(f"[cv_scorer] Error scoring {candidate.get('name', '?')}: {exc}")
            scored.append({
                **candidate,
                "scores": _zero_scores(error=str(exc)),
            })
    return scored


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str, label: str = "response") -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"[cv_scorer] Warning: could not parse {label}, using zero scores.")
        return _zero_scores(error=f"JSON parse failed: {raw[:200]}")


def _zero_scores(error: str = "") -> dict:
    """Return a zeroed-out score dict used when parsing or API calls fail."""
    return {
        "overall_score": 0,
        "must_have_score": 0,
        "nice_to_have_score": 0,
        "experience_score": 0,
        "keyword_score": 0,
        "strengths": [],
        "gaps": [error or "Scoring failed"],
        "recommendation": "Could not evaluate this candidate due to an error.",
    }
