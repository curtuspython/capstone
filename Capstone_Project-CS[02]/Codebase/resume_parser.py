"""
resume_parser.py
----------------
Parses candidate resumes from PDF, DOCX, or plain-text files and returns
structured candidate dictionaries containing both raw text and structured
data.

Parsing stack:
  - spaCy 3.x (en_core_web_sm) : Named-Entity Recognition for name/orgs,
                                   replaces pyresparser (broken on spaCy 3.x).
  - Regex                       : email, phone, years-of-experience extraction.
  - Keyword matching            : skill detection against a broad tech list.
  - pypdf                       : PDF text extraction.
  - python-docx                 : Word (.docx) text extraction.
  - Built-in open()             : plain-text (.txt / .md) reading.

Note on pyresparser: pyresparser targets spaCy 2.x and ships a config.cfg
that is incompatible with spaCy 3.x (raises E053). It is not used here;
structured extraction is implemented directly with spaCy 3.x NER + regex.
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
    Extract structured information from a resume using spaCy 3.x NER + regex.

    Replaces pyresparser (which is incompatible with spaCy 3.x) with a
    direct spaCy Named-Entity Recognition approach. Extracts:
      - name           : first PERSON entity found by NER
      - email          : regex
      - mobile_number  : regex
      - skills         : keyword match against _SKILL_KEYWORDS
      - education      : lines containing university/college/institute keywords
      - degree         : degree-level keywords found in text
      - experience     : lines containing job-title / experience section markers
      - total_experience: first N-year(s) pattern found (e.g. '5 years')

    Parameters
    ----------
    file_path : str
        Path to the resume (used only for filename-stem fallback name).
    text : str
       acted raw text of the resume. If empty, falls back to
        filename stem for name and empty lists for all other fields.

    Returns
    -------
    dict
        Structured resume data with consistent keys.
    """
    stem = Path(file_path).stem

    if not text.strip():
        return _empty_structured(stem)

    text_lower = text.lower()

    # -- Name via spaCy NER (PERSON entity) ----------------------------------
    name = _extract_name_ner(text) or stem

    # -- Email ----------------------------------------------------------------
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    email = email_match.group(0) if email_match else ""

    # -- Phone ----------------------------------------------------------------
    phone_match = re.search(
        r"(?:\+?\d[\d\s\-().]{7,14}\d)", text
    )
    mobile = phone_match.group(0).strip() if phone_match else ""

    # -- Skills (keyword intersection) ----------------------------------------
    skills = [
        kw for kw in _SKILL_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    ]

    # -- Degrees --------------------------------------------------------------
    degree = [
        kw.title() for kw in _DEGREE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    ]

    # -- Education lines ------------------------------------------------------
    edu_keywords = (
        "university", "college", "institute", "school", "academy",
        "b.sc", "m.sc", "b.tech", "m.tech", "bachelor", "master",
        "phd", "mba", "diploma",
    )
    education = [
        line.strip() for line in text.splitlines()
        if any(kw in line.lower() for kw in edu_keywords) and line.strip()
    ][:6]  # cap at 6 lines

    # -- Experience lines / sections ------------------------------------------
    exp_keywords = (
        "experience", "engineer", "developer", "analyst", "manager",
        "intern", "worked", "responsible", "led", "built", "designed",
    )
    experience = [
        line.strip() for line in text.splitlines()
        if any(kw in line.lower() for kw in exp_keywords) and line.strip()
    ][:8]  # cap at 8 lines

    # -- Total experience (e.g. '5 years of experience') ----------------------
    yrs_match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*year[s]?", text_lower
    )
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


def _extract_name_ner(text: str) -> str:
    """
    Use spaCy NER to find the first PERSON entity in the resume text.

    Looks only at the first 500 characters where the candidate's name
    typically appears (header / contact section).

    Returns an empty string if spaCy is unavailable or no PERSON found.
    """
    nlp = _get_nlp()
    if nlp is None:
        return ""
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""


def _empty_structured(name: str) -> dict:
    """Return a zeroed-out structured dict used as a safe fallback."""
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
