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
    --api-key  KEY  Groq API key (can also be set via GROQ_API_KEY env var)
    --output   DIR  Directory to write reports (default: current directory)
    --min-score N   Minimum composite score threshold, 0-100 (default: none)
    --help          Show this help message and exit

Examples
    # Using an environment variable for the API key:
    export GROQ_API_KEY=gsk_...
    python main.py --jd job_description.txt --cvs ./resumes/

    # Passing the API key as a CLI argument:
    python main.py --jd job_desc.pdf --cvs ./resumes/ --api-key gsk_... --output ./reports/

Two LLMs are used:
    LLM #1  llama-3.3-70b-versatile  — deep job-description analysis
    LLM #2  gemma2-9b-it             — per-CV scoring against requirements

All LLM calls are routed through the Groq API.
Get a free API key at https://console.groq.com/
"""

import argparse
import os
import sys

from groq import Groq

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
        description="CV Sorting using LLMs — AI-powered candidate ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --jd job_description.txt --cvs ./resumes/\n"
            "  python main.py --jd jd.pdf --cvs ./cvs/ --api-key gsk_... --output ./out/\n"
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
        help="Groq API key (falls back to GROQ_API_KEY environment variable)",
    )
    parser.add_argument(
        "--output", default=".", metavar="DIR",
        help="Output directory for ranking reports (default: current directory)",
    )
    parser.add_argument(
        "--min-score", type=float, default=None, metavar="N",
        help="Minimum composite score threshold 0–100 (flags candidates below threshold)",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(cli_key: str | None) -> str:
    """
    Resolve the Groq API key from CLI argument or environment variable.

    Precedence:
      1. --api-key CLI argument
      2. GROQ_API_KEY environment variable

    Raises SystemExit if no key is found.
    """
    key = cli_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        print(
            "[main] ERROR: Groq API key not found.\n"
            "  Set the GROQ_API_KEY environment variable:\n"
            "      export GROQ_API_KEY=gsk_...\n"
            "  Or pass it directly:\n"
            "      python main.py --jd ... --cvs ... --api-key gsk_...",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


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
┌──────────────────────────────────────────────────┐
│  CV Sorting using LLMs — Capstone Project CS[02]  │
│                                                    │
│  LLM #1 : llama-3.3-70b-versatile (JD analysis)   │
│  LLM #2 : gemma2-9b-it             (CV scoring)    │
└──────────────────────────────────────────────────┘
    """
    print(banner)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full CV sorting pipeline:
      1. Parse CLI arguments.
      2. Authenticate with Groq.
      3. Load and analyse the job description (LLM #1).
      4. Parse all candidate CVs from the given directory.
      5. Score each CV against the job requirements (LLM #2).
      6. Rank candidates by composite score.
      7. Print results to terminal and write reports to disk.
    """
    _print_banner()

    parser = _build_parser()
    args = parser.parse_args()

    # Step 1: Resolve API key and authenticate
    api_key = _resolve_api_key(args.api_key)
    client = Groq(api_key=api_key)
    print("[main] Groq client initialised.")

    # Step 2: Load job description
    print(f"\n[main] Loading job description from: {args.jd}")
    jd_text = _load_jd_text(args.jd)
    print(f"[main] Job description loaded ({len(jd_text)} characters).")

    # Step 3: Analyse JD with LLM #1 (llama-3.3-70b-versatile)
    print("\n[main] Step 1/4 — Analysing job description with LLM #1 …")
    requirements = analyze_job_description(jd_text, client)
    print(f"[main] Job title detected: {requirements.get('title', 'N/A')}")
    print(f"[main] Must-have skills : {len(requirements.get('must_have', []))} extracted")

    # Step 4: Parse candidate CVs
    print(f"\n[main] Step 2/4 — Parsing candidate CVs from: {args.cvs}")
    candidates = parse_resumes_from_directory(args.cvs)
    if not candidates:
        print("[main] No candidates found. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"[main] {len(candidates)} candidate(s) loaded.")

    # Step 5: Score each CV with LLM #2 (gemma2-9b-it)
    print(f"\n[main] Step 3/4 — Scoring {len(candidates)} CV(s) with LLM #2 …")
    scored_candidates = score_all_cvs(candidates, requirements, client)

    # Step 6: Rank candidates
    print("\n[main] Step 4/4 — Ranking candidates …")
    ranked_candidates = rank_candidates(scored_candidates, min_score=args.min_score)

    # Step 7: Print summary to terminal
    print(get_ranking_summary(ranked_candidates))

    # Step 8: Write reports to disk
    print("\n[main] Writing reports …")
    written_files = generate_report(
        ranked_candidates=ranked_candidates,
        requirements=requirements,
        output_dir=args.output,
    )

    print("\n[main] Done! Reports written:")
    for fmt, path in written_files.items():
        print(f"  [{fmt.upper()}] {path}")

    print("\n[main] ✔ CV Sorting complete.")


if __name__ == "__main__":
    main()
