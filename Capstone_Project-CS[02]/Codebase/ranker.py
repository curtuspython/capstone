"""
ranker.py
---------
Aggregates individual dimension scores into a final composite score and
produces a sorted, ranked list of candidates.

Weighting rationale:
  - must_have      : 40%  — mandatory requirements are the hardest filter
  - experience     : 25%  — experience depth is the second biggest differentiator
  - nice_to_have   : 20%  — preferred qualifications add value but are not blockers
  - keyword        : 15%  — keyword presence acts as a signal of domain fluency

The weights are normalised so they always sum to 1.0, making it easy to
adjust them without breaking the ranking logic.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Weighting configuration (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "must_have_score":    0.40,
    "experience_score":   0.25,
    "nice_to_have_score": 0.20,
    "keyword_score":      0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_composite_score(scores: dict) -> float:
    """
    Compute a single composite score from the multi-dimensional scoring dict.

    Parameters
    ----------
    scores : dict
        The 'scores' sub-dict from a scored candidate (output of cv_scorer).

    Returns
    -------
    float
        Composite score in the range [0.0, 100.0].
    """
    total = 0.0
    for key, weight in WEIGHTS.items():
        raw = scores.get(key, 0)
        # Ensure we have a numeric value even if the LLM returned a string
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        total += value * weight
    return round(total, 2)


def rank_candidates(
    scored_candidates: list[dict],
    min_score: Optional[float] = None,
) -> list[dict]:
    """
    Sort candidates by composite score (descending) and assign rank positions.

    Parameters
    ----------
    scored_candidates : list[dict]
        Candidates enriched with a 'scores' key (output of cv_scorer.score_all_cvs).
    min_score : float, optional
        If provided, candidates below this composite score are still included
        in the list but marked with ``"qualified": False``.

    Returns
    -------
    list[dict]
        Sorted list; each dict gains two new keys:
          ``composite_score`` (float) and ``rank`` (int, 1-based).
    """
    enriched: list[dict] = []

    for candidate in scored_candidates:
        scores = candidate.get("scores", {})
        composite = compute_composite_score(scores)
        qualified = True if min_score is None else composite >= min_score
        enriched.append({
            **candidate,
            "composite_score": composite,
            "qualified": qualified,
        })

    # Sort: qualified first (desc score), then disqualified (desc score)
    enriched.sort(key=lambda c: (c["qualified"], c["composite_score"]), reverse=True)

    # Assign 1-based ranks
    for idx, candidate in enumerate(enriched):
        candidate["rank"] = idx + 1

    return enriched


def get_ranking_summary(ranked_candidates: list[dict]) -> str:
    """
    Build a human-readable ranking table for terminal output.

    The 'Qualified' column is shown ONLY when a --min-score threshold was set
    and at least one candidate falls below it.  When every candidate passes
    (or no threshold was supplied) the column is hidden to keep the output
    clean — showing ✓ for every row would be meaningless noise.

    Parameters
    ----------
    ranked_candidates : list[dict]
        The output of rank_candidates().

    Returns
    -------
    str
        A formatted summary string.
    """
    # Only show the Qualified column if it is actually informative
    # (i.e. at least one candidate was flagged as not qualified)
    show_qualified = any(not c.get("qualified", True) for c in ranked_candidates)

    if show_qualified:
        header = f"{'Rank':<6}{'Candidate':<28}{'Composite':>10}  {'Overall':>8}  {'Qualified'}"
        separator = "=" * 72
        divider   = "-" * 72
    else:
        header = f"{'Rank':<6}{'Candidate':<28}{'Composite':>10}  {'Overall':>8}"
        separator = "=" * 58
        divider   = "-" * 58

    lines = ["\n" + separator, header, divider]

    for c in ranked_candidates:
        scores  = c.get("scores", {})
        overall = scores.get("overall_score", "N/A")
        row = (
            f"{c['rank']:<6}{c['name'][:27]:<28}"
            f"{c['composite_score']:>10.1f}  {str(overall):>8}"
        )
        if show_qualified:
            qualified_mark = "✓" if c.get("qualified", True) else "✘"
            row += f"  {qualified_mark}"
        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)
