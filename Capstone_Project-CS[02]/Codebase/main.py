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

Two LLMs are used:
    LLM #1  gemini-2.0-flash  - deep job-description analysis
    LLM #2  gemini-1.5-flash  - per-CV scoring against requirements

All LLM calls go through the Google Gemini API (free tier).
Get a free API key at https://aistudio.google.com/
"""

import argparse
import os
import sys
from pathlib import Path

import google.generativeai as genai

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


def _print_banner() -> None:
    """Print a welcome banner to stdout."""
    banner = """
+--------------------------------------------------+
|  CV Sorting using LLMs -- Capstone Project CS[02] |
|                                                    |
|  LLM #1 : gemini-2.0-flash  (JD analysis)         |
|  LLM #2 : gemini-1.5-flash  (CV scoring)          |
|  Provider: Google Gemini API (free tier)           |
+--------------------------------------------------+
    """
    print(banner)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full CV sorting pipeline:
      1. Parse CLI arguments.
      2. Resolve and configure Gemini API key.
      3. Load and analyse the job description (LLM #1 - gemini-2.0-flash).
      4. Parse all candidate CVs from the given directory.
      5. Score each CV against the job requirements (LLM #2 - gemini-1.5-flash).
      6. Rank candidates by composite score.
      7. Print results to terminal and write reports to disk.
    """
    _print_banner()

    parser = _build_parser()
    args = parser.parse_args()

    # Step 1: Resolve API key and configure Gemini globally
    api_key = _resolve_api_key(args.api_key)
    genai.configure(api_key=api_key)
    print("[main] Google Gemini API configured.")

    # Step 2: Load job description
    print(f"\n[main] Loading job description from: {args.jd}")
    jd_text = _load_jd_text(args.jd)
    print(f"[main] Job description loaded ({len(jd_text)} characters).")

    # Step 3: Analyse JD with LLM #1 (gemini-2.0-flash)
    print("\n[main] Step 1/4 -- Analysing job description with LLM #1 (gemini-2.0-flash) ...")
    requirements = analyze_job_description(jd_text)
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

    # Step 5: Score each CV with LLM #2 (gemini-1.5-flash)
    print(f"\n[main] Step 3/4 -- Scoring {len(candidates)} CV(s) with LLM #2 (gemini-1.5-flash) ...")
    scored_candidates = score_all_cvs(candidates, requirements)

    # Step 6: Rank candidates
    print("\n[main] Step 4/4 -- Ranking candidates ...")
    ranked_candidates = rank_candidates(scored_candidates, min_score=args.min_score)

    # Step 7: Print summary to terminal
    print(get_ranking_summary(ranked_candidates))

    # Step 8: Write reports to disk
    print("\n[main] Writing reports ...")
    written_files = generate_report(
        ranked_candidates=ranked_candidates,
        requirements=requirements,
        output_dir=args.output,
    )

    print("\n[main] Done! Reports written:")
    for fmt, path in written_files.items():
        print(f"  [{fmt.upper()}] {path}")

    print("\n[main] CV Sorting complete.")


if __name__ == "__main__":
    main()
