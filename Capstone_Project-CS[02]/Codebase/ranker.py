"""
ranker.py
---------
Aggregates individual dimension scores into a final composite score and
produces a sorted, ranked list of candidates.

Semantic scoring strategy (three-tier fallback -- first that succeeds wins):
  Tier 1 -- LlamaIndex GeminiEmbedding (llama-index-embeddings-gemini).
            LlamaIndex is the declared semantic matching framework (project
            requirement). get_llama_embed() from llm_client is called first.
            Note: at runtime this may return HTTP 404 because the
            llama-index-embeddings-gemini package targets the deprecated
            google.generativeai v1beta API endpoint which no longer serves
            text-embedding-004.  The error is caught and logged.
  Tier 2 -- Google Gemini text-embedding-004 via google-genai SDK directly.
            Tries two model-name formats (with/without 'models/' prefix)
            to handle SDK version differences.
  Tier 3 -- TF-IDF cosine similarity via scikit-learn (fully local, no API).
            Always produces real, differentiated scores without any network
            call or API key.
  Tier 4 -- Neutral default 50.0 (only if all three tiers above fail).

Weighting rationale (must sum to 1.0):
  - must_have      : 35%  -- mandatory requirements are the hardest filter
  - semantic       : 20%  -- embedding / TF-IDF similarity (meaning-level match)
  - experience     : 20%  -- experience depth is a key differentiator
  - nice_to_have   : 15%  -- preferred qualifications add value but not blockers
  - keyword        : 10%  -- keyword presence signals domain fluency
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
    Compute semantic similarity between each CV and the job description.

    Four-tier fallback (first tier that succeeds wins):

    Tier 1 -- LlamaIndex GeminiEmbedding (project requirement).
              Calls llm_client.get_llama_embed().get_text_embedding() to embed
              both the JD and each CV using the LlamaIndex framework, then
              computes cosine similarity.  May fail at runtime if the
              llama-index-embeddings-gemini package targets the deprecated
              v1beta API endpoint (returns HTTP 404).

    Tier 2 -- Google Gemini text-embedding-004 via google-genai SDK directly.
              Falls back here when Tier 1 raises a network or API error.
              Tries both 'models/text-embedding-004' and 'text-embedding-004'
              to handle SDK version differences.

    Tier 3 -- TF-IDF cosine similarity via scikit-learn (local, no API).
              Vectorises JD + CVs, computes cosine similarity.  Always
              produces real, differentiated scores without any network call.

    Tier 4 -- Neutral default 50.0 for all candidates.
              Only reached if scikit-learn is not installed.

    Parameters
    ----------
    candidates : list[dict]
        Scored candidate dicts (must contain 'text' key).
    jd_text : str
        Raw job description text.

    Returns
    -------
    list[dict]
        Same list with ``semantic_score`` (0-100 float) added to each entry.
    """
    # Tier 1: LlamaIndex
    if _try_llamaindex_embeddings(candidates, jd_text):
        return candidates

    print("[ranker] Tier 1 (LlamaIndex) unavailable -- trying Tier 2 (Gemini SDK).")

    # Tier 2: google-genai SDK
    if _try_gemini_embeddings(candidates, jd_text):
        return candidates

    print("[ranker] Tier 2 (Gemini SDK) unavailable -- trying Tier 3 (TF-IDF).")

    # Tier 3: TF-IDF
    if _try_tfidf_cosine(candidates, jd_text):
        return candidates

    # Tier 4: last resort
    print("[ranker] All semantic tiers failed -- using neutral default (50.0).")
    for candidate in candidates:
        candidate.setdefault("semantic_score", 50.0)
    return candidates


def _try_llamaindex_embeddings(candidates: list[dict], jd_text: str) -> bool:
    """
    Tier 1: embed texts using LlamaIndex GeminiEmbedding.

    Uses ``llm_client.get_llama_embed()`` which returns a
    ``llama_index.embeddings.gemini.GeminiEmbedding`` instance.
    Calls ``.get_text_embedding(text)`` on the LlamaIndex embed model to
    produce vector representations, then computes cosine similarity.

    Returns True on success, False on any error (API 404, import failure, etc.)
    """
    try:
        import llm_client  # noqa: PLC0415
        embed_model = llm_client.get_llama_embed()
        if embed_model is None:
            print("[ranker] LlamaIndex embed model not available (init failed).")
            return False

        # Embed the job description using LlamaIndex's embed interface
        jd_embedding = embed_model.get_text_embedding(jd_text[:4000])

        print("[ranker] Tier 1: LlamaIndex GeminiEmbedding active.")
        for candidate in candidates:
            cv_text = candidate.get("text", "")[:4000]
            cv_embedding = embed_model.get_text_embedding(cv_text)
            similarity = _cosine_similarity(jd_embedding, cv_embedding)
            score = round(max(0.0, min(100.0, similarity * 100)), 2)
            candidate["semantic_score"] = score
            print(
                f"[ranker] Semantic (LlamaIndex) -- "
                f"{candidate.get('name', '?')}: {score:.1f}"
            )
        return True

    except Exception as exc:
        print(f"[ranker] Tier 1 (LlamaIndex) error: {exc}")
        return False


def _try_gemini_embeddings(candidates: list[dict], jd_text: str) -> bool:
    """
    Tier 2: embed texts via Google Gemini text-embedding-004 directly.

    Uses ``llm_client.get_embed_client()`` -- a google-genai Client instance
    initialised with ``http_options={'api_version': 'v1'}`` -- to bypass the
    v1beta routing problem caused by LlamaIndex importing google.generativeai.
    text-embedding-004 is only available on the stable v1 API, not v1beta.

    Tries two model-name formats because different SDK versions accept different
    forms ('models/text-embedding-004' vs 'text-embedding-004').

    Returns True on success, False on any failure.
    """
    model_names = ["models/text-embedding-004", "text-embedding-004"]
    try:
        import llm_client  # noqa: PLC0415
        # Use the v1-pinned embed client to avoid LlamaIndex v1beta contamination
        client = llm_client.get_embed_client()

        jd_embedding = None
        used_model = None
        for model in model_names:
            try:
                resp = client.models.embed_content(
                    model=model, contents=jd_text[:4000]
                )
                jd_embedding = resp.embeddings[0].values
                used_model = model
                break
            except Exception as model_exc:
                print(f"[ranker] Tier 2: model '{model}' failed: {model_exc}")
                continue

        if jd_embedding is None:
            print("[ranker] Tier 2 (Gemini SDK): all model variants failed for JD embedding.")
            return False

        print(f"[ranker] Tier 2: Gemini SDK embedding active (model={used_model}).")
        raw_sims: list[float] = []
        cv_embeddings: list[list[float]] = []
        for candidate in candidates:
            cv_text = candidate.get("text", "")[:4000]
            resp = client.models.embed_content(model=used_model, contents=cv_text)
            cv_emb = resp.embeddings[0].values
            cv_embeddings.append(cv_emb)
            raw_sims.append(_cosine_similarity(jd_embedding, cv_emb))

        # Normalise to 0-100 (Gemini embeddings are unit-normalised so cosine
        # naturally sits in [0.6, 0.95] for relevant text pairs)
        for candidate, sim in zip(candidates, raw_sims):
            score = round(max(0.0, min(100.0, sim * 100)), 2)
            candidate["semantic_score"] = score
            print(
                f"[ranker] Semantic (Gemini) -- "
                f"{candidate.get('name', '?')}: {score:.1f}"
            )
        return True

    except Exception as exc:
        print(f"[ranker] Tier 2 (Gemini SDK) error: {exc}")
        return False


def _try_tfidf_cosine(candidates: list[dict], jd_text: str) -> bool:
    """
    Tier 3: TF-IDF vectoriser + cosine similarity (scikit-learn).

    Vectorises the job description and each CV in TF-IDF space, then
    computes cosine similarity between the JD vector and each CV vector.

    **Normalisation note:**
    Raw TF-IDF cosine values for JD-vs-CV pairs typically fall in [0.05, 0.35]
    because two real documents share vocabulary but are never identical.  A
    direct ``sim * 100`` mapping would produce scores of 5-35, which is
    systematically lower than LLM scores (0-100) and Gemini embedding scores
    (60-95).  Mixing these without normalisation biases the composite score.

    Solution: batch-relative min-max normalisation maps the worst match in
    the current batch to 10 and the best match to 100, preserving relative
    ranking while putting TF-IDF scores in the same range as other dimensions.
    If all candidates have equal similarity (degenerate case), every candidate
    receives a neutral score of 55.0.

    Returns True on success, False if scikit-learn is not installed.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.metrics.pairwise import cosine_similarity         # noqa: PLC0415

        cv_texts = [c.get("text", "") for c in candidates]
        corpus = [jd_text] + cv_texts

        vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        jd_vector  = tfidf_matrix[0]   # first row = job description
        cv_vectors = tfidf_matrix[1:]  # remaining rows = candidates

        raw_sims = cosine_similarity(jd_vector, cv_vectors)[0].tolist()

        # Batch min-max normalise to [10, 100] so TF-IDF scores are comparable
        # with LLM and embedding scores used in the weighted composite formula.
        min_sim = min(raw_sims)
        max_sim = max(raw_sims)
        score_range = max_sim - min_sim

        normalised: list[float] = []
        if score_range < 1e-9:
            # Degenerate case: all CVs have the same similarity to the JD
            normalised = [55.0] * len(raw_sims)
        else:
            for sim in raw_sims:
                # Scale to [10, 100]: worst gets 10, best gets 100
                norm = 10.0 + ((sim - min_sim) / score_range) * 90.0
                normalised.append(round(norm, 2))

        for candidate, score in zip(candidates, normalised):
            candidate["semantic_score"] = score
            print(
                f"[ranker] Semantic (TF-IDF, normalised) -- "
                f"{candidate.get('name', '?')}: {score:.1f}"
            )
        return True

    except ImportError:
        print(
            "[ranker] scikit-learn not installed; Tier 3 (TF-IDF) unavailable.\n"
            "  Fix: pip install scikit-learn>=1.3.0  (it is listed in requirements.txt)"
        )
        return False
    except Exception as exc:
        print(f"[ranker] Tier 3 (TF-IDF) error: {exc}")
        return False


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
