"""
report_generator.py
-------------------
Generates a detailed human-readable ranking report in two formats:
  1. Plain-text (.txt)  — always written; terminal-friendly
  2. CSV          (.csv) — always written; easy to import into Excel / Sheets

The report includes:
  - Job description summary and extracted requirements
  - Ranked candidate table with all dimension scores
  - Per-candidate strengths, gaps, and recruiter recommendation
  - Scoring methodology explanation
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    ranked_candidates: list[dict],
    requirements: dict,
    output_dir: str = ".",
    base_name: str = "cv_ranking_report",
    llm1: str = "gemini-2.5-flash",
    llm2: str = "gemini-2.5-flash",
) -> dict[str, str]:
    """
    Write the ranking report to disk in TXT and CSV formats.

    Parameters
    ----------
    ranked_candidates : list[dict]
        Output of ranker.rank_candidates().
    requirements : dict
        Structured JD analysis from jd_analyzer.analyze_job_description().
    output_dir : str
        Directory where the report files will be written.
    base_name : str
        Base filename (without extension) for the output files.
    llm1 : str
        Name of the model used for JD analysis (LLM #1).
    llm2 : str
        Name of the model used for CV scoring (LLM #2).

    Returns
    -------
    dict[str, str]
        Mapping from format name to the written file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: dict[str, str] = {}

    txt_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")
    _write_txt_report(txt_path, ranked_candidates, requirements, llm1=llm1, llm2=llm2)
    written["txt"] = txt_path
    print(f"[report_generator] Text report → {txt_path}")

    csv_path = os.path.join(output_dir, f"{base_name}_{timestamp}.csv")
    _write_csv_report(csv_path, ranked_candidates)
    written["csv"] = csv_path
    print(f"[report_generator] CSV report  → {csv_path}")

    return written


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_txt_report(
    path: str,
    ranked_candidates: list[dict],
    requirements: dict,
    llm1: str = "gemini-2.5-flash",
    llm2: str = "gemini-2.5-flash",
) -> None:
    """Write the full narrative text report."""
    lines: list[str] = []
    sep = "=" * 72
    thin = "-" * 72

    # ---- Header ----
    lines += [
        sep,
        "  CV SORTING REPORT  —  AI-Powered Candidate Ranking",
        f"  Generated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}",
        sep,
        "",
    ]

    # ---- Job summary ----
    lines += [
        "JOB DESCRIPTION SUMMARY",
        thin,
        f"Role        : {requirements.get('title', 'N/A')}",
        f"Min. Exp.   : {requirements.get('experience_min', 0)} year(s)",
        f"Summary     : {requirements.get('summary', 'N/A')}",
        "",
        "Must-Have Requirements:",
    ]
    for item in requirements.get("must_have", []):
        lines.append(f"  • {item}")
    lines.append("")
    lines.append("Nice-to-Have:")
    for item in requirements.get("nice_to_have", []):
        lines.append(f"  • {item}")
    lines += ["", "Keywords:  " + ", ".join(requirements.get("keywords", [])), ""]

    # ---- Ranking table ----
    lines += [
        sep,
        "CANDIDATE RANKING SUMMARY",
        thin,
        f"{'#':<4}{'Candidate':<26}{'Composite':>10}  {'MustHave':>9}  "
        f"{'NiceToHave':>11}  {'Exp':>5}  {'Keyword':>8}  {'Overall':>8}",
        thin,
    ]
    for c in ranked_candidates:
        s = c.get("scores", {})
        lines.append(
            f"{c['rank']:<4}{c['name'][:25]:<26}"
            f"{c['composite_score']:>10.1f}  "
            f"{_safe_int(s, 'must_have_score'):>9}  "
            f"{_safe_int(s, 'nice_to_have_score'):>11}  "
            f"{_safe_int(s, 'experience_score'):>5}  "
            f"{_safe_int(s, 'keyword_score'):>8}  "
            f"{_safe_int(s, 'overall_score'):>8}"
        )
    lines.append("")

    # ---- Per-candidate detail ----
    lines += [sep, "DETAILED CANDIDATE PROFILES", ""]
    for c in ranked_candidates:
        s = c.get("scores", {})
        lines += [
            thin,
            f"Rank #{c['rank']}  —  {c['name']}",
            f"File         : {c.get('file', 'N/A')}",
            f"Composite    : {c['composite_score']:.1f} / 100",
            f"Overall (LLM): {s.get('overall_score', 'N/A')} / 100",
            "",
            "Strengths:",
        ]
        for strength in s.get("strengths", []):
            lines.append(f"  + {strength}")
        lines.append("Gaps:")
        for gap in s.get("gaps", []):
            lines.append(f"  - {gap}")
        lines += [
            "",
            f"Recruiter Note: {s.get('recommendation', 'N/A')}",
            "",
        ]

    # ---- Methodology ----
    lines += [
        sep,
        "SCORING METHODOLOGY",
        thin,
        "This report uses two Large Language Models via the Google Gemini API:",
        f"  LLM #1 ({llm1}) — Analyses the job description and",
        "           extracts structured requirements (must-have, nice-to-have,",
        "           keywords, minimum experience). Runs once per session.",
        f"  LLM #2 ({llm2}) — Evaluates each CV against those requirements",
        "           and produces dimension-level scores plus narrative feedback.",
        "           Runs once per candidate in a loop.",
        "",
        "Composite score weights:",
        "  Must-Have      : 40%",
        "  Experience     : 25%",
        "  Nice-to-Have   : 20%",
        "  Keyword Match  : 15%",
        sep,
    ]

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_csv_report(path: str, ranked_candidates: list[dict]) -> None:
    """Write a flat CSV report for spreadsheet consumption."""
    fieldnames = [
        "rank", "name", "composite_score",
        "overall_score", "must_have_score", "nice_to_have_score",
        "experience_score", "keyword_score",
        "strengths", "gaps", "recommendation", "file",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for c in ranked_candidates:
            s = c.get("scores", {})
            writer.writerow({
                "rank":               c["rank"],
                "name":               c["name"],
                "composite_score":    c["composite_score"],
                "overall_score":      s.get("overall_score", 0),
                "must_have_score":    s.get("must_have_score", 0),
                "nice_to_have_score": s.get("nice_to_have_score", 0),
                "experience_score":   s.get("experience_score", 0),
                "keyword_score":      s.get("keyword_score", 0),
                "strengths":          " | ".join(s.get("strengths", [])),
                "gaps":               " | ".join(s.get("gaps", [])),
                "recommendation":     s.get("recommendation", ""),
                "file":               c.get("file", ""),
            })


def _safe_int(scores: dict, key: str) -> int:
    """Safely convert a score value to int, defaulting to 0."""
    try:
        return int(scores.get(key, 0))
    except (TypeError, ValueError):
        return 0
