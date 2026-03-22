"""
interactive.py
--------------
Terminal-based interactive refinement loop for the CV sorting pipeline.

Called from main.py when the user passes ``--interactive``.
Allows the recruiter to drill into results, tweak criteria, and re-run
scoring -- all without restarting the script.

Public API
----------
    run_interactive(scored, ranked, requirements, jd_text, output_dir, llm2, llm1)
        Enter the command loop.  Blocks until the user types 'quit'.
"""

import json

from cv_scorer import score_all_cvs
from ranker import rank_candidates, get_ranking_summary
from report_generator import save_report_to_file


# ---------------------------------------------------------------------------
# Help text (printed by the 'help' command)
# ---------------------------------------------------------------------------

_HELP = """
  Commands:
    show <rank>        Inspect match explanation & evidence for a candidate
    filter <skill>     Filter candidates whose CV contains the given skill
    reset              Remove all filters and show full ranking
    min-score <N>      Set minimum composite score threshold (0-100)
    edit-must          Edit must-have requirements (comma-separated)
    edit-nice          Edit nice-to-have requirements (comma-separated)
    edit-keywords      Edit keywords (comma-separated)
    rescore            Re-score ALL candidates with current requirements (expensive)
    rerank             Re-rank (no re-scoring) with current settings
    export             Save the current ranking to a TXT file on disk
    help               Show this help message
    quit               Exit interactive mode
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_interactive(
    scored_candidates: list[dict],
    ranked_candidates: list[dict],
    requirements: dict,
    jd_text: str,
    output_dir: str,
    llm2: str,
    llm1: str,
) -> None:
    """
    Terminal-based interactive refinement loop.

    Entered when the user passes ``--interactive``.  Maintains three mutable
    state variables across commands:

      current_scored  -- all candidates with LLM #2 scores; updated by 'rescore'
      current_ranked  -- current filtered/sorted view; rebuilt after each command
      current_reqs    -- live requirements dict; editable via edit-must/nice/keywords

    Parameters
    ----------
    scored_candidates : list[dict]
        Full scored candidate list from cv_scorer.score_all_cvs().
    ranked_candidates : list[dict]
        Initial ranked view from ranker.rank_candidates().
    requirements : dict
        Structured JD requirements from jd_analyzer (must_have, nice_to_have,
        keywords, experience_min, title, summary).
    jd_text : str
        Raw job description text (used for semantic re-ranking).
    output_dir : str
        Directory to write TXT reports on 'export'.
    llm2 : str
        Gemini model name for LLM #2 (used by 'rescore').
    llm1 : str
        Gemini model name for LLM #1 (passed through to report on 'export').
    """
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type 'help' for available commands, 'quit' to exit.")
    print("=" * 60)

    # Mutable session state
    current_scored = list(scored_candidates)
    current_ranked = list(ranked_candidates)
    current_reqs   = dict(requirements)
    min_score      = None       # composite score floor (float | None)
    active_filter  = None       # current skill filter string or None

    while True:
        try:
            cmd = input("\n[interactive] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[interactive] Exiting.")
            break

        if not cmd:
            continue

        parts  = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg    = parts[1].strip() if len(parts) > 1 else ""

        if action == "help":
            print(_HELP)

        elif action in ("quit", "exit", "q"):
            print("[interactive] Exiting interactive mode.")
            break

        elif action == "show":
            _cmd_show(current_ranked, arg)

        elif action == "filter":
            if not arg:
                print("[interactive] Usage: filter <skill>")
                continue
            active_filter = arg.lower()
            filtered = [
                c for c in current_scored
                if active_filter in c.get("text", "").lower()
                or active_filter in json.dumps(c.get("structured", {})).lower()
            ]
            print(f"[interactive] Filtered to {len(filtered)} candidate(s) matching '{arg}'.")
            current_ranked = rank_candidates(filtered, min_score=min_score, jd_text=jd_text)
            print(get_ranking_summary(current_ranked))

        elif action == "reset":
            active_filter  = None
            current_ranked = rank_candidates(current_scored, min_score=min_score, jd_text=jd_text)
            print("[interactive] Filters cleared. Full ranking restored.")
            print(get_ranking_summary(current_ranked))

        elif action == "min-score":
            try:
                val       = float(arg)
                min_score = val if val > 0 else None
                source    = _apply_filter(current_scored, active_filter)
                current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
                print(f"[interactive] Min-score set to {min_score}. Re-ranked.")
                print(get_ranking_summary(current_ranked))
            except ValueError:
                print("[interactive] Usage: min-score <number 0-100>")

        elif action == "edit-must":
            if not arg:
                print(f"[interactive] Current must-have: {current_reqs.get('must_have', [])}")
                print("[interactive] Usage: edit-must skill1, skill2, skill3")
                continue
            current_reqs["must_have"] = [s.strip() for s in arg.split(",") if s.strip()]
            print(f"[interactive] Must-have updated to: {current_reqs['must_have']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        elif action == "edit-nice":
            if not arg:
                print(f"[interactive] Current nice-to-have: {current_reqs.get('nice_to_have', [])}")
                print("[interactive] Usage: edit-nice skill1, skill2, skill3")
                continue
            current_reqs["nice_to_have"] = [s.strip() for s in arg.split(",") if s.strip()]
            print(f"[interactive] Nice-to-have updated to: {current_reqs['nice_to_have']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        elif action == "edit-keywords":
            if not arg:
                print(f"[interactive] Current keywords: {current_reqs.get('keywords', [])}")
                print("[interactive] Usage: edit-keywords kw1, kw2, kw3")
                continue
            current_reqs["keywords"] = [k.strip() for k in arg.split(",") if k.strip()]
            print(f"[interactive] Keywords updated to: {current_reqs['keywords']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        elif action == "rescore":
            # Expensive: one Gemini API call per candidate
            print("[interactive] Re-scoring all candidates with updated requirements ...")
            current_scored = score_all_cvs(current_scored, current_reqs, model=llm2)
            source         = _apply_filter(current_scored, active_filter)
            current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
            print(get_ranking_summary(current_ranked))

        elif action == "rerank":
            # Fast: no API calls, just re-sort with current filter/threshold
            source         = _apply_filter(current_scored, active_filter)
            current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
            print("[interactive] Re-ranked.")
            print(get_ranking_summary(current_ranked))

        elif action == "export":
            # Only time a file is written; path goes to --output directory
            txt_path = save_report_to_file(
                ranked_candidates=current_ranked,
                requirements=current_reqs,
                output_dir=output_dir,
                llm1=llm1,
                llm2=llm2,
            )
            print(f"[interactive] Report saved -> {txt_path}")

        else:
            print(f"[interactive] Unknown command: '{action}'. Type 'help' for options.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_filter(candidates: list[dict], skill_filter: str | None) -> list[dict]:
    """
    Return the subset of candidates matching the active skill filter.
    If no filter is set, returns the full list unchanged.

    Parameters
    ----------
    candidates : list[dict]
        Full scored candidate list.
    skill_filter : str | None
        Lowercase substring to match, or None to return all.
    """
    if not skill_filter:
        return candidates
    return [
        c for c in candidates
        if skill_filter in c.get("text", "").lower()
        or skill_filter in json.dumps(c.get("structured", {})).lower()
    ]


def _cmd_show(ranked: list[dict], arg: str) -> None:
    """
    Print the full match breakdown for a candidate by rank number.

    Shows composite score, semantic score, overall LLM score, all five
    dimension scores (must-have, nice-to-have, experience, keywords, overall),
    LLM-generated strengths and gaps, recruiter recommendation, and
    structured profile fields from pyresparser (Tier 1) or spaCy NER (Tier 2).

    Parameters
    ----------
    ranked : list[dict]
        Current ranked candidate list.
    arg : str
        Rank number as a string (e.g. "1").
    """
    try:
        rank_num = int(arg)
    except (ValueError, TypeError):
        print("[interactive] Usage: show <rank_number>")
        return

    # Locate the candidate matching the requested rank
    candidate = next((c for c in ranked if c.get("rank") == rank_num), None)
    if candidate is None:
        print(f"[interactive] No candidate at rank #{rank_num}.")
        return

    scores     = candidate.get("scores", {})
    structured = candidate.get("structured", {})
    sep        = "-" * 60

    print(f"\n{sep}")
    print(f"  RANK #{candidate['rank']}  --  {candidate['name']}")
    print(sep)
    print(f"  File           : {candidate.get('file', 'N/A')}")
    print(f"  Composite Score: {candidate.get('composite_score', 'N/A')} / 100")
    print(f"  Semantic Score : {candidate.get('semantic_score', 'N/A')} / 100"
          "  (LlamaIndex -> Gemini -> TF-IDF)")
    print(f"  Overall (LLM)  : {scores.get('overall_score', 'N/A')} / 100")
    print()
    print(f"  Must-Have      : {scores.get('must_have_score', 'N/A')}")
    print(f"  Nice-to-Have   : {scores.get('nice_to_have_score', 'N/A')}")
    print(f"  Experience     : {scores.get('experience_score', 'N/A')}")
    print(f"  Keywords       : {scores.get('keyword_score', 'N/A')}")
    print()

    print("  Strengths (supporting evidence):")
    for s in scores.get("strengths", []):
        print(f"    + {s}")

    print("  Gaps (areas of concern):")
    for g in scores.get("gaps", []):
        print(f"    - {g}")
    print()

    print(f"  Recruiter Note: {scores.get('recommendation', 'N/A')}")

    # Structured profile extracted by pyresparser Tier 1 or spaCy NER Tier 2
    profile_keys = ["skills", "education", "degree", "total_experience"]
    if structured and any(structured.get(k) for k in profile_keys):
        print()
        print("  Structured Profile (pyresparser Tier 1 / spaCy NER Tier 2):")
        if structured.get("skills"):
            print(f"    Skills     : {', '.join(structured['skills'][:15])}")
        if structured.get("education"):
            print(f"    Education  : {', '.join(str(e) for e in structured['education'][:5])}")
        if structured.get("degree"):
            print(f"    Degree     : {', '.join(str(d) for d in structured['degree'][:5])}")
        if structured.get("total_experience"):
            print(f"    Experience : {structured['total_experience']} years")

    print(sep)
