"""
report_generator.py
-------------------
Builds the candidate ranking report and outputs it to the terminal.

Output strategy
---------------
- During a normal pipeline run the full report is printed directly to
  the terminal (stdout).  No CSV or TXT files are created automatically.
- In interactive mode the 'export' command lets the recruiter persist the
  current (possibly filtered / re-scored) ranking to a TXT file on disk
  by calling save_report_to_file().

Report sections
---------------
  1. Job description summary and extracted requirements
  2. Ranked candidate summary table  -- Composite, Semantic, Overall scores
  3. Per-candidate detail profiles   -- all five dimension scores, strengths,
                                        gaps, and recruiter recommendation note
  4. Scoring methodology             -- weights, LLM choices, fallback chain
"""

import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_report(
    ranked_candidates: list[dict],
    requirements: dict,
    llm1: str = "gemini-2.5-flash",
    llm2: str = "gemini-2.5-pro",
) -> None:
    """
    Print the full ranking report directly to the terminal (stdout).

    This is the primary output method -- no files are written.  The report
    includes the JD summary, full ranked table, per-candidate detail, and
    the scoring methodology section.

    Parameters
    ----------
    ranked_candidates : list[dict]
        Output of ranker.rank_candidates().
    requirements : dict
        Structured JD analysis from jd_analyzer.analyze_job_description().
    llm1 : str
        Name of the model used for JD analysis (LLM #1).
    llm2 : str
        Name of the model used for CV scoring (LLM #2).
    """
    lines = _build_report_lines(ranked_candidates, requirements, llm1=llm1, llm2=llm2)
    print("\n".join(lines))


def save_report_to_file(
    ranked_candidates: list[dict],
    requirements: dict,
    output_dir: str = ".",
    base_name: str = "cv_ranking_report",
    llm1: str = "gemini-2.5-flash",
    llm2: str = "gemini-2.5-pro",
) -> str:
    """
    Save the ranking report to a TXT file on disk and return its path.

    Only called by the interactive mode 'export' command -- not during
    the normal pipeline run.  Useful when the recruiter has refined
    criteria or filtered candidates and wants to persist the result.

    Parameters
    ----------
    ranked_candidates : list[dict]
        Output of ranker.rank_candidates() (possibly after re-scoring).
    requirements : dict
        Current requirements dict (may differ from the original if edited
        via 'edit-must' / 'edit-nice' / 'edit-keywords' in interactive mode).
    output_dir : str
        Directory to write the file in (created if it does not exist).
    base_name : str
        Filename prefix; a timestamp is appended automatically.
    llm1 : str
        Model name for LLM #1 (shown in the methodology section).
    llm2 : str
        Model name for LLM #2 (shown in the methodology section).

    Returns
    -------
    str
        Absolute path to the written TXT file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")

    lines = _build_report_lines(ranked_candidates, requirements, llm1=llm1, llm2=llm2)
    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_generator] Report saved to: {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_report_lines(
    ranked_candidates: list[dict],
    requirements: dict,
    llm1: str = "gemini-2.5-flash",
    llm2: str = "gemini-2.5-pro",
) -> list[str]:
    """
    Build the full report as a list of plain-text lines.

    Shared by print_report() and save_report_to_file() so the output is
    always identical whether it is shown in the terminal or saved to disk.

    Parameters
    ----------
    ranked_candidates : list[dict]
    requirements : dict
    llm1 : str
    llm2 : str

    Returns
    -------
    list[str]
        One string per output line (no trailing newlines on individual items).
    """
    lines: list[str] = []  # accumulate one string per output line
    sep  = "=" * 72          # heavy separator for major sections
    thin = "-" * 72          # light separator for sub-sections

    # ---- Report header with title and generation timestamp ----
    lines += [
        sep,
        "  CV SORTING REPORT  --  AI-Powered Candidate Ranking",
        f"  Generated : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}",
        sep,
        "",
    ]

    # ---- Section 1: Job description summary ----
    # Shows the role title, minimum experience, and all extracted requirements
    # from the LLM #1 analysis so the recruiter can verify what was parsed.
    lines += [
        "JOB DESCRIPTION SUMMARY",
        thin,
        f"Role      : {requirements.get('title', 'N/A')}",
        f"Min. Exp. : {requirements.get('experience_min', 0)} year(s)",
        f"Summary   : {requirements.get('summary', 'N/A')}",
        "",
        "Must-Have Requirements:",
    ]
    for item in requirements.get("must_have", []):
        lines.append(f"  * {item}")

    lines.append("")
    lines.append("Nice-to-Have:")
    for item in requirements.get("nice_to_have", []):
        lines.append(f"  * {item}")

    lines += [
        "",
        "Keywords : " + ", ".join(requirements.get("keywords", [])),
        "",
    ]

    # ---- Section 2: Ranked candidate summary table ----
    # Compact table showing Composite, Semantic, and Overall scores.
    # Candidates below the --min-score threshold are flagged.
    lines += [
        sep,
        "CANDIDATE RANKING SUMMARY",
        thin,
        # Column header (Composite = weighted blend, Semantic = TF-IDF/embedding,
        # Overall = LLM #2's holistic score)
        f"{'#':<4}{'Candidate':<26}{'Composite':>10}  {'Semantic':>9}  {'Overall':>8}",
        thin,
    ]
    for c in ranked_candidates:
        s = c.get("scores", {})
        # Append a flag if this candidate is below the minimum score threshold
        flag = "  [below threshold]" if not c.get("qualified", True) else ""
        lines.append(
            f"{c['rank']:<4}{c['name'][:25]:<26}"
            f"{c['composite_score']:>10.1f}  "
            f"{str(round(c.get('semantic_score', 0), 1)):>9}  "
            f"{_safe_int(s, 'overall_score'):>8}"
            f"{flag}"
        )
    lines.append("")

    # ---- Section 3: Per-candidate detail profiles ----
    # Full breakdown for each candidate: scores, strengths, gaps, and
    # the recruiter recommendation note generated by LLM #2.
    lines += [sep, "DETAILED CANDIDATE PROFILES", ""]
    for c in ranked_candidates:
        s = c.get("scores", {})
        # Candidate header with rank, name, composite, semantic, and overall scores
        lines += [
            thin,
            f"Rank #{c['rank']}  --  {c['name']}",
            f"File           : {c.get('file', 'N/A')}",
            f"Composite Score: {c['composite_score']:.1f} / 100",
            f"Semantic Score : {round(c.get('semantic_score', 0), 1)} / 100  "
            "(LlamaIndex -> Gemini SDK -> TF-IDF fallback)",
            f"Overall (LLM)  : {s.get('overall_score', 'N/A')} / 100",
            "",
            # Individual dimension scores from LLM #2 (all five dimensions)
            f"Must-Have      : {s.get('must_have_score', 'N/A')} / 100",
            f"Nice-to-Have   : {s.get('nice_to_have_score', 'N/A')} / 100",
            f"Experience     : {s.get('experience_score', 'N/A')} / 100",
            f"Keywords       : {s.get('keyword_score', 'N/A')} / 100",
            "",
            "Strengths:",  # LLM-identified supporting evidence
        ]
        for strength in s.get("strengths", []):
            lines.append(f"  + {strength}")
        lines.append("Gaps:")  # LLM-identified areas of concern
        for gap in s.get("gaps", []):
            lines.append(f"  - {gap}")
        lines += [
            "",
            f"Recruiter Note : {s.get('recommendation', 'N/A')}",
            "",
        ]

    # ---- Section 4: Scoring methodology ----
    # Documents the full technical approach for transparency and reproducibility.
    lines += [
        sep,
        "SCORING METHODOLOGY",
        thin,
        "Two Large Language Models via the Google Gemini API, orchestrated",
        "by LangChain with LlamaIndex for semantic matching:",
        "",
        f"  LLM #1 ({llm1})",
        "    Analyses the job description once and extracts structured",
        "    requirements (must-have, nice-to-have, keywords, min experience).",
        "    Uses a LangChain ChatPromptTemplate -> JsonOutputParser chain.",
        "",
        f"  LLM #2 ({llm2})",
        "    Evaluates each CV against those requirements and produces",
        "    dimension-level scores plus narrative feedback (strengths,",
        "    gaps, recruiter note).  Uses LangChain chain per candidate.",
        "",
        "  Semantic Scoring (4-tier fallback -- first that succeeds wins):",
        "    Tier 1  LlamaIndex GeminiEmbedding   (cosine similarity)",
        "    Tier 2  Gemini SDK text-embedding-004 (v1 endpoint, isolated client)",
        "    Tier 3  TF-IDF cosine similarity      (scikit-learn, local)",
        "    Tier 4  Neutral default 50.0          (if all above fail)",
        "    TF-IDF scores are batch-normalised to [10, 100] to match the",
        "    range of embedding-based scores in the composite formula.",
        "",
        "  Resume Extraction (2-tier fallback):",
        "    Tier 1  pyresparser  (attempted first; may raise E053 on spaCy 3.x)",
        "    Tier 2  spaCy 3.x NER + regex  (automatic fallback)",
        "",
        "  Composite Score Weights (must sum to 100%):",
        "    Must-Have      35%",
        "    Semantic Match 20%  (LlamaIndex / TF-IDF fallback)",
        "    Experience     20%",
        "    Nice-to-Have   15%",
        "    Keyword Match  10%",
        sep,
    ]

    return lines


def _safe_int(scores: dict, key: str) -> int:
    """
    Safely cast a score value to int, returning 0 on any failure.

    Used in the summary table to convert LLM-returned score values
    (which may be strings, floats, or None) to clean integers.

    Parameters
    ----------
    scores : dict
        The 'scores' sub-dict from a scored candidate.
    key : str
        The score key to look up (e.g. 'overall_score').

    Returns
    -------
    int
        The score as an integer, or 0 if conversion fails.
    """
    try:
        return int(scores.get(key, 0))
    except (TypeError, ValueError):
        return 0
