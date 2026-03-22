"""
resume_parser.py
----------------
Parses candidate resumes from PDF, DOCX, or plain-text files and returns
structured candidate dictionaries containing both raw text and structured
data.

Structured extraction strategy (two-tier fallback):

  Tier 1 — pyresparser (project requirement).
            pyresparser is attempted first on every CV.  It is, however,
            incompatible with spaCy 3.x: it ships a config.cfg in the old
            spaCy 2.x format which raises [E053] on spaCy 3.x.  If this
            error (or any other import/runtime failure) is detected, the
            module falls through to Tier 2 automatically.

  Tier 2 — Custom spaCy 3.x NER + regex extractor (always available).
            Extracts the same fields as pyresparser (name, email, phone,
            skills, education, degree, experience, total_experience) using
            spaCy 3.x Named-Entity Recognition and lightweight regex
            patterns.  Works on Python 3.11+ with any spaCy 3.x model.

File format support:
  - pypdf       : PDF text extraction
  - python-docx : Word (.docx) text extraction
  - Built-in    : plain-text (.txt / .md) reading
"""

import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None  # type: ignore

# ---------------------------------------------------------------------------
# Tier 1 — pyresparser (project requirement, attempted first).
# pyresparser targets spaCy 2.x; its shipped config.cfg is incompatible
# with spaCy 3.x and raises [E053].  We catch that and any other import
# failure here so the fallback to Tier 2 is transparent.
# ---------------------------------------------------------------------------
_HAS_PYRESPARSER = False
try:
    # NLTK data needed by pyresparser
    import nltk as _nltk
    for _r in ("stopwords", "punkt", "averaged_perceptron_tagger", "punkt_tab"):
        _nltk.download(_r, quiet=True)
    from pyresparser import ResumeParser as _PyResParser  # noqa: PLC0415
    # Smoke-test: pyresparser registers a custom spaCy component at import
    # time.  If [E053] is going to fire it fires here.
    _HAS_PYRESPARSER = True
    print("[resume_parser] pyresparser loaded successfully (Tier 1 active).")
except Exception as _e:
    print(
        f"[resume_parser] pyresparser unavailable ({type(_e).__name__}: {_e}) "
        "\u2014 using spaCy 3.x NER fallback (Tier 2)."
    )

# ---------------------------------------------------------------------------
# spaCy NLP model — loaded once at module level for performance.
# Auto-downloads en_core_web_sm on first run if the model is missing.
# ---------------------------------------------------------------------------
_NLP = None  # lazy-loaded in _get_nlp()


def _get_nlp():
    """Return the shared spaCy NLP model, loading/downloading on first call."""
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy  # noqa: PLC0415
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            print("[resume_parser] Downloading spaCy model 'en_core_web_sm' (one-time) ...")
            from spacy.cli import download as _dl  # noqa: PLC0415
            _dl("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
        return _NLP
    except Exception as exc:  # pragma: no cover
        print(f"[resume_parser] spaCy unavailable ({exc}); NER extraction disabled.")
        return None


# ---------------------------------------------------------------------------
# Common tech/professional skills keyword list for matching against CV text.
# Kept intentionally broad — LLM #2 does the nuanced matching; this is a
# lightweight signal for structured data context.
# ---------------------------------------------------------------------------
_SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "swift", "kotlin", "scala", "r", "matlab", "bash", "sql",
    # Web / backend
    "fastapi", "django", "flask", "spring", "express", "node.js", "nodejs",
    "react", "vue", "angular", "html", "css", "rest", "graphql", "grpc",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "llm", "tensorflow", "pytorch",
    "keras", "scikit-learn", "pandas", "numpy", "spark", "hadoop",
    "langchain", "llamaindex", "openai", "gemini",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "jenkins", "github actions", "linux",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    "cassandra", "bigquery", "snowflake",
    # General
    "git", "agile", "scrum", "microservices", "api",
]

_DEGREE_KEYWORDS = [
    "bachelor", "b.sc", "b.s.", "b.e.", "b.tech",
    "master", "m.sc", "m.s.", "m.e.", "m.tech", "mba",
    "phd", "ph.d", "doctorate",
    "associate", "diploma",
]


# ---------------------------------------------------------------------------
# Name-extraction helpers  (used by _ner_person)
# ---------------------------------------------------------------------------

# Words that appear in CV headers right after a name but are NOT part of it.
# e.g. "DAVID CHEN Email" → strip "Email" → "DAVID CHEN"
_HEADER_NOISE = frozenset({
    "email", "phone", "mobile", "tel", "linkedin", "github", "twitter",
    "instagram", "address", "contact", "website", "portfolio", "resume",
    "cv", "curriculum", "vitae", "profile", "objective", "summary", "name",
    "at", "the", "page",
})


def _looks_like_name(text: str) -> bool:
    """
    Return True if ``text`` looks like a human name.

    Rules:
      - 2 to 4 space-separated words
      - Every word contains only letters, hyphens, or apostrophes  (no dots,
        no digits -- rules out things like 'Vue.js' or 'Python3')
      - At least two words must be longer than 1 character
        (rules out initials-only like 'A B')
      - No word may be a known header-noise term (Email, Phone, LinkedIn, ...)
    """
    parts = text.strip().split()
    if not (2 <= len(parts) <= 4):
        return False
    if not all(re.match(r"^[A-Za-z][A-Za-z'\-]*$", p) for p in parts):
        return False
    # Require at least 2 words longer than 1 char to avoid 'A B' style initials
    if sum(1 for p in parts if len(p) > 1) < 2:
        return False
    # Reject if any word is a known header-noise label (e.g. 'Email', 'Phone')
    if any(p.lower() in _HEADER_NOISE for p in parts):
        return False
    return True


def _clean_ner_name(raw: str) -> str:
    """
    Strip trailing header-noise words from a spaCy NER-extracted name.

    e.g. ``'DAVID CHEN Email'`` → ``'DAVID CHEN'``
    """
    words = raw.split()
    while words and words[-1].lower().rstrip(":.|") in _HEADER_NOISE:
        words.pop()
    return " ".join(words)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """
    Extract raw text from a single resume file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the resume.

    Returns
    -------
    str
        The extracted plain text, or an empty string on failure.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".txt", ".md"):
        return _parse_txt(path)
    else:
        print(f"[resume_parser] Unsupported file format: {suffix} — skipping {path.name}")
        return ""


def extract_structured_data(file_path: str, text: str = "") -> dict:
    """
    Extract structured information from a resume.

    Tier 1 — pyresparser (project requirement):
        Uses pyresparser.ResumeParser which internally leverages NLTK and
        spaCy for NLP-based field extraction.  Attempted first on every CV.
        Falls through to Tier 2 when pyresparser is unavailable (e.g. spaCy
        version mismatch) or raises any runtime error.

    Tier 2 — Custom spaCy 3.x NER + regex:
        Extracts name (PERSON entity), email, phone (regex), skills
        (keyword match), education/degree (keyword lines), experience
        (job-title line detection), and years of experience (regex).
        Compatible with Python 3.11+ and spaCy 3.x.

    Parameters
    ----------
    file_path : str
        Path to the resume file.
    text : str
        Pre-extracted raw text (passed in to avoid re-reading the file).

    Returns
    -------
    dict
        Structured resume data with consistent keys:
        name, email, mobile_number, skills, education, experience,
        total_experience, degree.
    """
    # --- Tier 1: pyresparser ---
    if _HAS_PYRESPARSER:
        try:
            data = _PyResParser(file_path).get_extracted_data()
            return {
                "name":             data.get("name", "") or Path(file_path).stem,
                "email":            data.get("email", ""),
                "mobile_number":    data.get("mobile_number", ""),
                "skills":           data.get("skills", []) or [],
                "education":        data.get("education", []) or [],
                "experience":       data.get("experience", []) or [],
                "total_experience": data.get("total_experience", 0) or 0,
                "degree":           data.get("degree", []) or [],
            }
        except Exception as exc:
            print(
                f"[resume_parser] pyresparser Tier 1 failed for "
                f"{Path(file_path).name}: {exc} — falling back to Tier 2."
            )

    # --- Tier 2: spaCy 3.x NER + regex ---
    return _extract_with_spacy_ner(file_path, text)


def _extract_with_spacy_ner(file_path: str, text: str) -> dict:
    """
    Tier 2 structured extractor using spaCy 3.x NER + regex.

    This is the fallback when pyresparser (Tier 1) is unavailable or fails.
    Extracts the same fields pyresparser would produce so downstream code
    receives a consistent dict regardless of which tier ran.
    """
    stem = Path(file_path).stem

    if not text.strip():
        return _empty_structured(stem)

    text_lower = text.lower()

    # -- Name: use three-strategy extractor; fall back to filename stem -------
    name = _ner_person(text)
    if not name:
        # Last resort: use the filename stem but clean underscores/dashes
        name = Path(file_path).stem.replace("_", " ").replace("-", " ").title()
        print(
            f"[resume_parser] Name not found in header; using filename: '{name}'"
        )

    # -- Email ----------------------------------------------------------------
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    email = email_match.group(0) if email_match else ""

    # -- Phone ----------------------------------------------------------------
    phone_match = re.search(r"(?:\+?\d[\d\s\-().]{7,14}\d)", text)
    mobile = phone_match.group(0).strip() if phone_match else ""

    # -- Skills: keyword intersection against a broad tech list ---------------
    skills = [
        kw for kw in _SKILL_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    ]

    # -- Degrees --------------------------------------------------------------
    degree = [
        kw.title() for kw in _DEGREE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    ]

    # -- Education lines (lines containing institution/degree keywords) --------
    edu_kw = (
        "university", "college", "institute", "school", "academy",
        "b.sc", "m.sc", "b.tech", "m.tech", "bachelor", "master",
        "phd", "mba", "diploma",
    )
    education = [
        line.strip() for line in text.splitlines()
        if any(kw in line.lower() for kw in edu_kw) and line.strip()
    ][:6]  # cap at 6 lines to keep the dict lean

    # -- Experience lines (job-title / action-verb bearing lines) -------------
    exp_kw = (
        "experience", "engineer", "developer", "analyst", "manager",
        "intern", "worked", "responsible", "led", "built", "designed",
    )
    experience = [
        line.strip() for line in text.splitlines()
        if any(kw in line.lower() for kw in exp_kw) and line.strip()
    ][:8]  # cap at 8 lines

    # -- Total years of experience (e.g. '5 years', '3+ years') ---------------
    yrs_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*year[s]?", text_lower)
    total_experience = float(yrs_match.group(1)) if yrs_match else 0

    return {
        "name":             name,
        "email":            email,
        "mobile_number":    mobile,
        "skills":           skills,
        "education":        education,
        "experience":       experience,
        "total_experience": total_experience,
        "degree":           degree,
    }


def _ner_person(text: str) -> str:
    """
    Extract a clean candidate name from the top of a resume.

    Three strategies applied in order (first that succeeds wins):

    Strategy 1 -- First non-empty line heuristic.
        Most well-formatted CVs put the candidate's name on the very first
        line.  If the first non-empty line passes ``_looks_like_name`` it is
        returned immediately (title-cased).  This handles ALL-CAPS headers
        like ``ALICE JOHNSON`` without needing NER at all.

    Strategy 2 -- spaCy NER on header with noise cleaning.
        Run spaCy NER on the first 600 characters.  For each PERSON entity
        found, strip trailing header-noise words (Email, Phone, LinkedIn,…)
        using ``_clean_ner_name``, then validate with ``_looks_like_name``.
        Filters out false positives such as framework names (``Vue.js``) or
        label bleed (``DAVID CHEN Email``).

    Strategy 3 -- Scan first 10 lines for any name-like line.
        Fallback scan for cases where the name is not on line 1 but appears
        within the first few lines of the document header.

    Returns
    -------
    str
        A cleaned, title-cased name string, or ``''`` if nothing matched.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Strategy 1: first non-empty line looks like a name?
    if lines and _looks_like_name(lines[0]):
        return lines[0].title()

    # Strategy 2: spaCy NER with noise stripping on header block
    nlp = _get_nlp()
    if nlp is not None:
        header = "\n".join(lines[:12])[:600]  # top of document
        doc = nlp(header)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned = _clean_ner_name(ent.text.strip())
                if cleaned and _looks_like_name(cleaned):
                    return cleaned.title()

    # Strategy 3: scan first 10 lines for any name-like line
    for line in lines[:10]:
        # Strip common label prefixes ("Name: Alice Johnson")
        candidate = re.sub(
            r"^(?:name|candidate|applicant)\s*[:\-]\s*",
            "", line, flags=re.IGNORECASE,
        ).strip()
        if _looks_like_name(candidate):
            return candidate.title()

    return ""


def _empty_structured(name: str) -> dict:
    """Return a zeroed-out structured dict — safe fallback for empty text."""
    return {
        "name":             name,
        "email":            "",
        "mobile_number":    "",
        "skills":           [],
        "education":        [],
        "experience":       [],
        "total_experience": 0,
        "degree":           [],
    }


def parse_resumes_from_directory(directory: str) -> list[dict]:
    """
    Scan a directory for resume files and return a list of candidate dicts.

    Each dict has the shape::

        {
            "name":       str,   # from spaCy NER or filename stem
            "file":       str,   # full path
            "text":       str,   # raw extracted text
            "structured": dict,  # structured fields from spaCy NER
        }

    Parameters
    ----------
    directory : str
        Path to the folder containing resumes.

    Returns
    -------
    list[dict]
        One entry per successfully parsed resume.
    """
    supported_exts = {".pdf", ".docx", ".txt", ".md"}
    candidates: list[dict] = []

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {directory}")

    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_exts
    )

    if not files:
        print(f"[resume_parser] No supported resume files found in '{directory}'.")
        return candidates

    for file_path in files:
        print(f"[resume_parser] Parsing: {file_path.name}")
        text = extract_text_from_file(str(file_path))
        if text.strip():
            structured = extract_structured_data(str(file_path), text=text)
            candidate_name = structured.get("name") or file_path.stem
            candidates.append({
                "name": candidate_name,
                "file": str(file_path),
                "text": text.strip(),
                "structured": structured,
            })
        else:
            print(f"[resume_parser] Warning: no text extracted from {file_path.name}")

    print(f"[resume_parser] Loaded {len(candidates)} resume(s) from '{directory}'.")
    return candidates


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Run: pip install pypdf")
    try:
        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    except Exception as exc:
        print(f"[resume_parser] PDF parse error on '{path.name}': {exc}")
        return ""


def _parse_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    if DocxDocument is None:
        raise ImportError("python-docx is not installed. Run: pip install python-docx")
    try:
        doc = DocxDocument(str(path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as exc:
        print(f"[resume_parser] DOCX parse error on '{path.name}': {exc}")
        return ""


def _parse_txt(path: Path) -> str:
    """Read a plain-text file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[resume_parser] TXT read error on '{path.name}': {exc}")
        return ""
