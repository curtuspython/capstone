"""
main.py
-------
Entry point for the CV Sorting system (Capstone Project CS[02]).

Usage
-----
    python main.py --jd <job_description_file> --cvs <cv_directory> [OPTIONS]

Required Arguments
    --jd   PATH   Path to the job description file (.txt / .pdf / .docx)
    --cvs  PATH   Path to the directory containing candidate CV files

Optional Arguments
    --api-key  KEY  Google Gemini API key
                    (can also be set via GEMINI_API_KEY env var
                     or stored in a .env file in the same directory)
    --output   DIR  Directory to write reports (default: current directory)
    --min-score N   Minimum composite score threshold, 0-100 (default: none)
    --help          Show this help message and exit

Examples
    # Using an environment variable for the API key:
    export GEMINI_API_KEY=AIza...
    python main.py --jd job_description.txt --cvs ./resumes/

    # Passing the API key as a CLI argument:
    python main.py --jd jd.pdf --cvs ./resumes/ --api-key AIza... --output ./reports/

Two LLMs are used by default (overridable via --llm1 / --llm2):
    LLM #1  gemini-2.5-flash - structured JD extraction (runs once, moderate complexity)
    LLM #2  gemini-2.5-pro   - deep CV scoring with judgment (runs per candidate, high complexity)

All LLM calls go through the Google Gemini API.
Get a free API key at https://aistudio.google.com/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import llm_client

from resume_parser import parse_resumes_from_directory, extract_text_from_file
from jd_analyzer import analyze_job_description
from cv_scorer import score_all_cvs
from ranker import rank_candidates, get_ranking_summary
from report_generator import generate_report


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CV Sorting using LLMs - AI-powered candidate ranking (Google Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --jd job_description.txt --cvs ./resumes/\n"
            "  python main.py --jd jd.pdf --cvs ./cvs/ --api-key AIza... --output ./out/\n"
        ),
    )
    parser.add_argument(
        "--jd", required=True, metavar="PATH",
        help="Path to the job description file (.txt, .pdf, or .docx)",
    )
    parser.add_argument(
        "--cvs", required=True, metavar="PATH",
        help="Directory containing candidate CV files (.pdf, .docx, or .txt)",
    )
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="Google Gemini API key (falls back to GEMINI_API_KEY env var or .env file)",
    )
    parser.add_argument(
        "--output", default=".", metavar="DIR",
        help="Output directory for ranking reports (default: current directory)",
    )
    parser.add_argument(
        "--min-score", type=float, default=None, metavar="N",
        help="Minimum composite score threshold 0-100 (flags candidates below threshold)",
    )
    parser.add_argument(
        "--llm1", default="gemini-2.5-flash", metavar="MODEL",
        help="Gemini model for JD analysis / LLM #1 (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--llm2", default="gemini-2.5-pro", metavar="MODEL",
        help="Gemini model for CV scoring / LLM #2 (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive mode after ranking to refine criteria, "
             "filter candidates, inspect match details, and re-run sorting",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(cli_key: str | None) -> str:
    """
    Resolve the Gemini API key with the following precedence:
      1. --api-key CLI argument
      2. GEMINI_API_KEY environment variable
      3. GEMINI_API_KEY entry inside a .env file in the Codebase directory

    Raises SystemExit if no key is found.
    """
    # 1. CLI argument
    if cli_key:
        return cli_key

    # 2. Environment variable
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key

    # 3. .env file sitting next to main.py
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY") and "=" in line:
                key = line.split("=", 1)[1].strip()
                if key:
                    print("[main] API key loaded from .env file.")
                    return key

    print(
        "[main] ERROR: Gemini API key not found.\n"
        "  Option 1 - Set environment variable:\n"
        "      export GEMINI_API_KEY=AIza...\n"
        "  Option 2 - Pass via CLI:\n"
        "      python main.py --jd ... --cvs ... --api-key AIza...\n"
        "  Option 3 - Add to .env file in the Codebase directory:\n"
        "      GEMINI_API_KEY=AIza...\n"
        "\n"
        "  Get a FREE key at: https://aistudio.google.com/",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_jd_text(jd_path: str) -> str:
    """
    Load the job description as plain text from any supported file format.

    Parameters
    ----------
    jd_path : str
        Path to the job description file.

    Returns
    -------
    str
        Extracted plain text of the job description.
    """
    if not os.path.isfile(jd_path):
        print(f"[main] ERROR: Job description file not found: {jd_path}", file=sys.stderr)
        sys.exit(1)

    text = extract_text_from_file(jd_path)
    if not text.strip():
        print(f"[main] ERROR: No text could be extracted from: {jd_path}", file=sys.stderr)
        sys.exit(1)

    return text.strip()


def _print_banner(llm1: str, llm2: str) -> None:
    """Print a welcome banner showing the configured models."""
    banner = f"""
+----------------------------------------------------+
|  CV Sorting using LLMs -- Capstone Project CS[02]  |
|                                                    |
|  LLM #1 : {llm1:<40}|
|  LLM #2 : {llm2:<40}|
|  Provider: Google Gemini API                       |
+----------------------------------------------------+
    """
    print(banner)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full CV sorting pipeline:
      1. Parse CLI arguments (including optional --llm1 / --llm2 model overrides).
      2. Resolve and configure Gemini API key.
      3. Load and analyse the job description (LLM #1).
      4. Parse all candidate CVs from the given directory.
      5. Score each CV against the job requirements (LLM #2, per-candidate loop).
      6. Rank candidates by composite score.
      7. Print results to terminal and write reports to disk.
    """
    # Parse args FIRST so the banner can display the actual chosen model names
    parser = _build_parser()
    args = parser.parse_args()

    _print_banner(llm1=args.llm1, llm2=args.llm2)

    # Step 1: Resolve API key and initialise shared Gemini client
    api_key = _resolve_api_key(args.api_key)
    llm_client.init(api_key)
    print("[main] Google Gemini API configured.")

    # Step 2: Load job description
    print(f"\n[main] Loading job description from: {args.jd}")
    jd_text = _load_jd_text(args.jd)
    print(f"[main] Job description loaded ({len(jd_text)} characters).")

    # Step 3: Analyse JD with LLM #1 (model chosen via --llm1)
    print(f"\n[main] Step 1/4 -- Analysing job description with LLM #1 ({args.llm1}) ...")
    requirements = analyze_job_description(jd_text, model=args.llm1)
    print(f"[main] Job title detected : {requirements.get('title', 'N/A')}")
    print(f"[main] Must-have skills   : {len(requirements.get('must_have', []))} extracted")
    print(f"[main] Keywords           : {len(requirements.get('keywords', []))} extracted")

    # Step 4: Parse candidate CVs
    print(f"\n[main] Step 2/4 -- Parsing candidate CVs from: {args.cvs}")
    candidates = parse_resumes_from_directory(args.cvs)
    if not candidates:
        print("[main] No candidates found. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"[main] {len(candidates)} candidate(s) loaded.")

    # Step 5: Score each CV with LLM #2 (model chosen via --llm2)
    print(f"\n[main] Step 3/4 -- Scoring {len(candidates)} CV(s) with LLM #2 ({args.llm2}) ...")
    scored_candidates = score_all_cvs(candidates, requirements, model=args.llm2)

    # Step 6: Rank candidates (includes LlamaIndex semantic similarity scoring)
    print("\n[main] Step 4/4 -- Ranking candidates (+ LlamaIndex semantic matching) ...")
    ranked_candidates = rank_candidates(scored_candidates, min_score=args.min_score, jd_text=jd_text)

    # Step 7: Print summary to terminal
    print(get_ranking_summary(ranked_candidates))

    # Step 8: Write reports to disk
    print("\n[main] Writing reports ...")
    written_files = generate_report(
        ranked_candidates=ranked_candidates,
        requirements=requirements,
        output_dir=args.output,
        llm1=args.llm1,
        llm2=args.llm2,
    )

    print("\n[main] Done! Reports written:")
    for fmt, path in written_files.items():
        print(f"  [{fmt.upper()}] {path}")

    # Step 9: Interactive query refinement mode (if requested)
    if args.interactive:
        _interactive_loop(
            scored_candidates=scored_candidates,
            ranked_candidates=ranked_candidates,
            requirements=requirements,
            jd_text=jd_text,
            output_dir=args.output,
            llm2=args.llm2,
            llm1=args.llm1,
        )

    print("\n[main] CV Sorting complete.")


# ---------------------------------------------------------------------------
# Interactive terminal mode  (Step 4 – query refinement without GUI)
# ---------------------------------------------------------------------------

_INTERACTIVE_HELP = """
  Commands:
    show <rank>        Inspect match explanation & evidence for a candidate
    filter <skill>     Filter candidates whose CV contains the given skill
    reset              Remove all filters and show full ranking
    min-score <N>      Set minimum composite score threshold (0-100)
    edit-must          Edit must-have requirements (comma-separated)
    edit-nice          Edit nice-to-have requirements (comma-separated)
    edit-keywords      Edit keywords (comma-separated)
    rescore            Re-score filtered candidates with current requirements
    rerank             Re-rank (no re-scoring) with current settings
    export             Write TXT + CSV reports for the current ranking
    help               Show this help message
    quit               Exit interactive mode
"""


def _interactive_loop(
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

    Allows the recruiter to inspect match explanations, filter by skill,
    adjust job criteria, and re-run sorting — all from the terminal,
    with no GUI dependency (rule 8g).
    """
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type 'help' for available commands, 'quit' to exit.")
    print("=" * 60)

    current_scored = list(scored_candidates)
    current_ranked = list(ranked_candidates)
    current_reqs = dict(requirements)
    min_score = None
    active_filter = None

    while True:
        try:
            cmd = input("\n[interactive] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[interactive] Exiting.")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # ---- help ----
        if action == "help":
            print(_INTERACTIVE_HELP)

        # ---- quit ----
        elif action in ("quit", "exit", "q"):
            print("[interactive] Exiting interactive mode.")
            break

        # ---- show <rank> : inspect match explanation ----
        elif action == "show":
            _cmd_show(current_ranked, arg)

        # ---- filter <skill> ----
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

        # ---- reset ----
        elif action == "reset":
            active_filter = None
            current_ranked = rank_candidates(current_scored, min_score=min_score, jd_text=jd_text)
            print("[interactive] Filters cleared. Full ranking restored.")
            print(get_ranking_summary(current_ranked))

        # ---- min-score <N> ----
        elif action == "min-score":
            try:
                val = float(arg)
                min_score = val if val > 0 else None
                source = current_scored if not active_filter else [
                    c for c in current_scored
                    if active_filter in c.get("text", "").lower()
                ]
                current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
                print(f"[interactive] Min-score set to {min_score}. Re-ranked.")
                print(get_ranking_summary(current_ranked))
            except ValueError:
                print("[interactive] Usage: min-score <number 0-100>")

        # ---- edit-must ----
        elif action == "edit-must":
            if not arg:
                print(f"[interactive] Current must-have: {current_reqs.get('must_have', [])}")
                print("[interactive] Usage: edit-must skill1, skill2, skill3")
                continue
            current_reqs["must_have"] = [s.strip() for s in arg.split(",") if s.strip()]
            print(f"[interactive] Must-have updated to: {current_reqs['must_have']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        # ---- edit-nice ----
        elif action == "edit-nice":
            if not arg:
                print(f"[interactive] Current nice-to-have: {current_reqs.get('nice_to_have', [])}")
                print("[interactive] Usage: edit-nice skill1, skill2, skill3")
                continue
            current_reqs["nice_to_have"] = [s.strip() for s in arg.split(",") if s.strip()]
            print(f"[interactive] Nice-to-have updated to: {current_reqs['nice_to_have']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        # ---- edit-keywords ----
        elif action == "edit-keywords":
            if not arg:
                print(f"[interactive] Current keywords: {current_reqs.get('keywords', [])}")
                print("[interactive] Usage: edit-keywords kw1, kw2, kw3")
                continue
            current_reqs["keywords"] = [k.strip() for k in arg.split(",") if k.strip()]
            print(f"[interactive] Keywords updated to: {current_reqs['keywords']}")
            print("[interactive] Run 'rescore' to re-score with updated requirements.")

        # ---- rescore : re-score with updated requirements ----
        elif action == "rescore":
            print("[interactive] Re-scoring all candidates with updated requirements ...")
            current_scored = score_all_cvs(current_scored, current_reqs, model=llm2)
            source = current_scored if not active_filter else [
                c for c in current_scored
                if active_filter in c.get("text", "").lower()
            ]
            current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
            print(get_ranking_summary(current_ranked))

        # ---- rerank : re-rank without re-scoring ----
        elif action == "rerank":
            source = current_scored if not active_filter else [
                c for c in current_scored
                if active_filter in c.get("text", "").lower()
            ]
            current_ranked = rank_candidates(source, min_score=min_score, jd_text=jd_text)
            print("[interactive] Re-ranked.")
            print(get_ranking_summary(current_ranked))

        # ---- export ----
        elif action == "export":
            written = generate_report(
                ranked_candidates=current_ranked,
                requirements=current_reqs,
                output_dir=output_dir,
                llm1=llm1,
                llm2=llm2,
            )
            for fmt, path in written.items():
                print(f"  [{fmt.upper()}] {path}")

        else:
            print(f"[interactive] Unknown command: '{action}'. Type 'help' for options.")


def _cmd_show(ranked: list[dict], arg: str) -> None:
    """
    Display detailed match explanation and evidence for a candidate by rank.

    Prints dimension-level scores, strengths (supporting evidence), gaps
    (areas of concern), recruiter recommendation, and structured profile
    data extracted by pyresparser — all in the terminal.
    """
    try:
        rank_num = int(arg)
    except (ValueError, TypeError):
        print("[interactive] Usage: show <rank_number>")
        return

    candidate = None
    for c in ranked:
        if c.get("rank") == rank_num:
            candidate = c
            break

    if candidate is None:
        print(f"[interactive] No candidate at rank #{rank_num}.")
        return

    scores = candidate.get("scores", {})
    structured = candidate.get("structured", {})
    sep = "-" * 60

    print(f"\n{sep}")
    print(f"  RANK #{candidate['rank']}  —  {candidate['name']}")
    print(sep)
    print(f"  File           : {candidate.get('file', 'N/A')}")
    print(f"  Composite Score: {candidate.get('composite_score', 'N/A')} / 100")
    print(f"  Semantic Score : {candidate.get('semantic_score', 'N/A')} / 100  (LlamaIndex)")
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

    # Show pyresparser structured data if available
    if structured and any(structured.get(k) for k in ["skills", "education", "degree"]):
        print()
        print("  Structured Profile (pyresparser):")
        if structured.get("skills"):
            print(f"    Skills     : {', '.join(structured['skills'][:15])}")
        if structured.get("education"):
            print(f"    Education  : {', '.join(str(e) for e in structured['education'][:5])}")
        if structured.get("degree"):
            print(f"    Degree     : {', '.join(str(d) for d in structured['degree'][:5])}")
        if structured.get("total_experience"):
            print(f"    Experience : {structured['total_experience']} years")

    print(sep)


if __name__ == "__main__":
    main()
