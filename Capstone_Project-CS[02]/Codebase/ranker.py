"""
ranker.py
---------
Aggregates individual dimension scores into a final composite score and
produces a sorted, ranked list of candidates.

Uses **LlamaIndex** with Google Gemini embeddings for semantic candidate-job
matching.  The cosine similarity between each CV's embedding and the job
description embedding is converted to a 0-100 ``semantic_score`` that
becomes the fifth dimension in the composite formula.

Weighting rationale (must sum to 1.0):
  - must_have      : 35%  — mandatory requirements are the hardest filter
  - semantic       : 20%  — LlamaIndex embedding similarity (deep meaning match)
  - experience     : 20%  — experience depth is a key differentiator
  - nice_to_have   : 15%  — preferred qualifications add value but not blockers
  - keyword        : 10%  — keyword presence acts as a signal of domain fluency
"""

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Weighting configuration (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "must_have_score":    0.35,
    "semantic_score":     0.20,
    "experience_score":   0.20,
    "nice_to_have_score": 0.15,
    "keyword_score":      0.10,
}


# ---------------------------------------------------------------------------
# LlamaIndex semantic similarity
# ---------------------------------------------------------------------------

def compute_semantic_scores(candidates: list[dict], jd_text: str) -> list[dict]:
    """
    Use LlamaIndex GeminiEmbedding to compute semantic similarity between
    each candidate CV and the job description.

    The cosine similarity (range -1 to 1) is scaled to 0-100 and stored
    as ``candidate["semantic_score"]``.

    Parameters
    ----------
    candidates : list[dict]
        Scored candidate dicts (must contain 'text' key).
    jd_text : str
        Raw job description text.

    Returns
    -------
    list[dict]
        Same list, each candidate now has a ``semantic_score`` key.
    """
    try:
        import llm_client
        embed_model = llm_client.get_llama_embed()

        # Embed the job description once
        jd_embedding = embed_model.get_text_embedding(jd_text[:2000])

        for candidate in candidates:
            cv_text = candidate.get("text", "")[:2000]
            cv_embedding = embed_model.get_text_embedding(cv_text)
            similarity = _cosine_similarity(jd_embedding, cv_embedding)
            # Scale cosine similarity (0-1 range for positive embeddings) to 0-100
            candidate["semantic_score"] = round(max(0.0, min(100.0, similarity * 100)), 2)
            print(f"[ranker] Semantic score for {candidate.get('name', '?')}: "
                  f"{candidate['semantic_score']:.1f}")

    except Exception as exc:
        print(f"[ranker] LlamaIndex semantic scoring unavailable: {exc}")
        print("[ranker] Using neutral default (50.0) for semantic_score.")
        for candidate in candidates:
            candidate["semantic_score"] = 50.0

    return candidates


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_composite_score(scores: dict, semantic_score: float = 50.0) -> float:
    """
    Compute a single composite score from the multi-dimensional scoring dict.

    Parameters
    ----------
    scores : dict
        The 'scores' sub-dict from a scored candidate (output of cv_scorer).
    semantic_score : float
        LlamaIndex semantic similarity score (0-100).

    Returns
    -------
    float
        Composite score in the range [0.0, 100.0].
    """
    total = 0.0
    for key, weight in WEIGHTS.items():
        if key == "semantic_score":
            raw = semantic_score
        else:
            raw = scores.get(key, 0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        total += value * weight
    return round(total, 2)


def rank_candidates(
    scored_candidates: list[dict],
    min_score: Optional[float] = None,
    jd_text: str = "",
) -> list[dict]:
    """
    Sort candidates by composite score (descending) and assign rank positions.

    If ``jd_text`` is provided, LlamaIndex semantic similarity scores are
    computed and folded into the composite score.

    Parameters
    ----------
    scored_candidates : list[dict]
        Candidates enriched with a 'scores' key (output of cv_scorer.score_all_cvs).
    min_score : float, optional
        If provided, candidates below this composite score are still included
        in the list but marked with ``"qualified": False``.
    jd_text : str, optional
        Raw JD text for LlamaIndex semantic similarity computation.

    Returns
    -------
    list[dict]
        Sorted list; each dict gains keys:
          ``composite_score``, ``semantic_score``, ``rank``, ``qualified``.
    """
    # Compute LlamaIndex semantic similarity scores if JD text is available
    if jd_text:
        scored_candidates = compute_semantic_scores(scored_candidates, jd_text)

    enriched: list[dict] = []

    for candidate in scored_candidates:
        scores = candidate.get("scores", {})
        semantic = candidate.get("semantic_score", 50.0)
        composite = compute_composite_score(scores, semantic)
        qualified = True if min_score is None else composite >= min_score
        enriched.append({
            **candidate,
            "composite_score": composite,
            "semantic_score": semantic,
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
        header = (f"{'Rank':<6}{'Candidate':<24}{'Composite':>10}  "
                  f"{'Semantic':>9}  {'Overall':>8}  {'Qualified'}")
        separator = "=" * 78
        divider   = "-" * 78
    else:
        header = (f"{'Rank':<6}{'Candidate':<24}{'Composite':>10}  "
                  f"{'Semantic':>9}  {'Overall':>8}")
        separator = "=" * 64
        divider   = "-" * 64

    lines = ["\n" + separator, header, divider]

    for c in ranked_candidates:
        scores  = c.get("scores", {})
        overall = scores.get("overall_score", "N/A")
        semantic = c.get("semantic_score", "N/A")
        row = (
            f"{c['rank']:<6}{c['name'][:23]:<24}"
            f"{c['composite_score']:>10.1f}  "
            f"{str(semantic):>9}  {str(overall):>8}"
        )
        if show_qualified:
            qualified_mark = "✓" if c.get("qualified", True) else "✘"
            row += f"  {qualified_mark}"
        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)
