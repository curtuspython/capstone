"""
cv_scorer.py
------------
Uses LLM #2 (Google Gemini: gemini-1.5-flash) to score a single candidate
CV against the structured job requirements extracted by jd_analyzer.py.

For each CV the LLM returns:
  - overall_score      : int  0-100
  - must_have_score    : int  0-100  (match against mandatory requirements)
  - nice_to_have_score : int  0-100  (match against preferred qualifications)
  - experience_score   : int  0-100  (relevant experience fit)
  - keyword_score      : int  0-100  (keyword presence in CV)
  - strengths          : list[str] (top 3 candidate strengths)
  - gaps               : list[str] (top 3 missing areas)
  - recommendation     : str  (1-2 sentence recruiter note)

Using gemini-1.5-flash (lighter/faster) here keeps costs low while still
providing meaningful per-candidate scoring. The separation from
jd_analyzer lets us run scoring in a loop without re-running the heavier
gemini-2.0-flash model for every single CV.
"""

import json
import re

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM #2 - lighter, faster model used for per-CV scoring loop
SCORER_MODEL = "gemini-1.5-flash"

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

def score_cv(candidate: dict, requirements: dict) -> dict:
    """
    Score a single candidate CV against the extracted job requirements.

    Requires google.generativeai to be configured with an API key
    before calling (via genai.configure in main.py).

    Parameters
    ----------
    candidate : dict
        Must contain keys 'name', 'file', and 'text' (raw CV text).
    requirements : dict
        Structured JD requirements from jd_analyzer.analyze_job_description().

    Returns
    -------
    dict
        The input candidate dict enriched with a 'scores' key containing
        the full scoring breakdown from the LLM.
    """
    candidate_name = candidate.get("name", "Unknown")
    print(f"[cv_scorer] Scoring: {candidate_name} (model: {SCORER_MODEL})")

    model = genai.GenerativeModel(
        model_name=SCORER_MODEL,
        system_instruction=_SYSTEM_INSTRUCTION,
    )

    # Truncate CV text to avoid exceeding context length
    cv_text_trunc = candidate["text"][:6000]
    requirements_json = json.dumps(requirements, indent=2)

    prompt = _USER_PROMPT_TEMPLATE.format(
        requirements_json=requirements_json,
        cv_text=cv_text_trunc,
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.15,
            max_output_tokens=768,
        ),
    )

    raw_output = response.text.strip()
    scores = _parse_json_response(raw_output, label=f"scores for {candidate_name}")

    # Return a new dict so original candidate is not mutated
    return {**candidate, "scores": scores}


def score_all_cvs(candidates: list[dict], requirements: dict) -> list[dict]:
    """
    Score every candidate in the list and return the enriched list.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dicts from resume_parser.parse_resumes_from_directory().
    requirements : dict
        Structured JD requirements from jd_analyzer.analyze_job_description().

    Returns
    -------
    list[dict]
        Each dict now includes a 'scores' key with numeric scores and narrative.
    """
    scored: list[dict] = []
    for candidate in candidates:
        try:
            scored.append(score_cv(candidate, requirements))
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
