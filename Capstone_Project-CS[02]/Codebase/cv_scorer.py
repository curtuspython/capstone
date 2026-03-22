"""
cv_scorer.py
------------
Uses **LangChain** with Google Gemini (LLM #2) to score a single candidate
CV against the structured job requirements extracted by jd_analyzer.py (LLM #1).

LangChain serves as the orchestrator for:
  - Prompt templating (ChatPromptTemplate) — takes structured resume data
    (from spaCy NER) and parsed job description as input.
  - LLM invocation (ChatGoogleGenerativeAI via langchain-google-genai)
  - Structured JSON output parsing (JsonOutputParser) — scores per requirement
    plus an overall fit score with explanation.

For each CV the LLM returns:
  - overall_score      : int  0-100
  - must_have_score    : int  0-100  (match against mandatory requirements)
  - nice_to_have_score : int  0-100  (match against preferred qualifications)
  - experience_score   : int  0-100  (relevant experience fit)
  - keyword_score      : int  0-100  (keyword presence in CV)
  - strengths          : list[str] (top 3 candidate strengths)
  - gaps               : list[str] (top 3 missing areas)
  - recommendation     : str  (1-2 sentence recruiter note)

LLM #1 (gemini-2.5-flash) -- fast structured JD extraction, runs once per session.
LLM #2 (gemini-2.5-pro)   -- deep CV assessment with nuanced judgment, runs per candidate.
The Pro model is deliberately placed here because CV scoring is the high-complexity,
high-stakes task: it must infer skills from experience, compare two documents, and
produce calibrated scores that directly determine the final hiring ranking.
"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import llm_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM #2 - gemini-2.5-pro: the most capable Gemini model, placed here because
# CV scoring is the highest-complexity task in the pipeline. The model must
# read a full CV, infer skills from described experience, compare against job
# requirements, and produce calibrated numeric scores that determine the final
# ranking. Accuracy here matters most -- wrong scores = wrong hiring decision.
SCORER_MODEL = "gemini-2.5-pro"

_SYSTEM_INSTRUCTION = (
    "You are a senior technical recruiter. "
    "Evaluate candidate resumes against structured job requirements and "
    "provide objective, numeric scores with brief justifications."
)

_USER_PROMPT = """
Score the following candidate resume against the job requirements.

Job Requirements (JSON):
{requirements_json}

Candidate Structured Profile (extracted via pyresparser Tier 1 / spaCy NER Tier 2):
{structured_data}

Candidate CV (Full Text):
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

    Uses LangChain to orchestrate the prompt → LLM → JSON parser chain.
    The prompt template takes both the structured resume data (from
    pyresparser Tier 1 / spaCy NER Tier 2) and the parsed job description
    as input, and asks the LLM to score the match for each requirement and
    generate an overall fit score with an explanation.

    Parameters
    ----------
    candidate : dict
        Must contain keys 'name', 'file', 'text', and optionally 'structured'.
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
    print(f"[cv_scorer] Scoring: {candidate_name} (LangChain + {model})")

    # Step 1: Retrieve a LangChain-wrapped Gemini LLM from the shared client
    # module.  Temperature is set to 1.0 for nuanced, calibrated scoring.
    llm = llm_client.get_langchain_llm(model=model, temperature=1.0)

    # Step 2: Build the LangChain chain.
    # The chain flows:  prompt template -> Gemini LLM -> JSON output parser.
    # This produces a structured dict of scores directly from the LLM response.
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_INSTRUCTION),
        ("human", _USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # Step 3: Prepare the structured resume data (from pyresparser Tier 1 or
    # spaCy NER Tier 2) as a JSON string for the prompt context.
    structured = candidate.get("structured", {})
    structured_str = json.dumps(structured, indent=2) if structured else "N/A"

    # Step 4: Invoke the chain with all three inputs (JD, structured profile,
    # raw CV text).  CV text is truncated to 6000 chars to stay within
    # the LLM's context window.
    try:
        scores = chain.invoke({
            "requirements_json": json.dumps(requirements, indent=2),
            "structured_data": structured_str,
            "cv_text": candidate["text"][:6000],
        })
        print(f"[cv_scorer] LangChain chain completed for {candidate_name}.")
    except Exception as exc:
        # LangChain chain failed (network, parsing, etc.) — fall back to
        # calling the Gemini API directly without LangChain orchestration.
        print(f"[cv_scorer] LangChain failed for {candidate_name}, trying direct API: {exc}")
        scores = _fallback_direct_api(candidate, requirements, model)

    # Return a NEW dict (spread operator) so the original candidate is not mutated
    return {**candidate, "scores": scores}


def score_all_cvs(candidates: list[dict], requirements: dict, model: str = SCORER_MODEL) -> list[dict]:
    """
    Score every candidate in the list and return the enriched list.

    Results are collected into a list with detailed per-candidate feedback
    including scores, strengths, gaps, and recommendations.

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
    scored: list[dict] = []  # accumulate scored candidate dicts
    for candidate in candidates:
        try:
            # Score each candidate individually; append the enriched dict
            scored.append(score_cv(candidate, requirements, model=model))
        except Exception as exc:
            # On any failure, assign zero scores and log the error so the
            # pipeline continues with remaining candidates.
            print(f"[cv_scorer] Error scoring {candidate.get('name', '?')}: {exc}")
            scored.append({
                **candidate,
                "scores": _zero_scores(error=str(exc)),
            })
    return scored


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fallback_direct_api(candidate: dict, requirements: dict, model: str) -> dict:
    """
    Fallback: call the Gemini API directly if the LangChain chain fails.

    Bypasses LangChain entirely and sends the prompt directly to the
    google-genai SDK client.  The response is parsed as JSON manually.

    Parameters
    ----------
    candidate : dict
        The candidate dict (must contain 'text' and optionally 'structured').
    requirements : dict
        Structured JD requirements.
    model : str
        Gemini model name.

    Returns
    -------
    dict
        Parsed scoring dict (same schema as the LangChain chain output).
    """
    from google.genai import types

    # Use the default genai.Client (v1beta) for generate_content
    client = llm_client.get()
    structured = candidate.get("structured", {})
    structured_str = json.dumps(structured, indent=2) if structured else "N/A"

    # Format the prompt template with actual values
    prompt = _USER_PROMPT.format(
        requirements_json=json.dumps(requirements, indent=2),
        structured_data=structured_str,
        cv_text=candidate["text"][:6000],
    )

    # Call the Gemini API directly (no LangChain)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=1.0,
            max_output_tokens=16000,
        ),
    )

    # Parse the raw text response as JSON
    raw_output = response.text.strip()
    return _parse_json_response(raw_output, label=f"scores for {candidate.get('name', '?')}")


def _parse_json_response(raw: str, label: str = "response") -> dict:
    """
    Parse JSON from an LLM response, stripping markdown fences if present.

    LLMs often wrap JSON in ```json ... ``` code fences.  This function
    strips those fences, then attempts json.loads().  If that fails, it
    tries to extract the first {...} block as a last resort.

    Parameters
    ----------
    raw : str
        Raw LLM response text.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    dict
        Parsed JSON dict, or a zeroed-out score dict on failure.
    """
    # Strip markdown code fences (```json ... ```) if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract the first JSON object {...} from the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # All parsing attempts failed — return zeroed scores
        print(f"[cv_scorer] Warning: could not parse {label}, using zero scores.")
        return _zero_scores(error=f"JSON parse failed: {raw[:200]}")


def _zero_scores(error: str = "") -> dict:
    """
    Return a zeroed-out score dict used when scoring fails.

    Ensures downstream code (ranker, report_generator) always receives
    a consistent dict shape regardless of whether the LLM call succeeded.
    The error message is stored in the 'gaps' list for visibility.
    """
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
