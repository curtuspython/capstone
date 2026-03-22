"""
resume_parser.py
----------------
Parses candidate resumes from PDF, DOCX, or plain-text files and returns
structured candidate dictionaries containing both raw text and structured
data extracted via pyresparser.

Parsing stack:
  - pyresparser   : structured information extraction (name, email, skills,
                     education, experience) as required by the project spec.
  - pypdf          : PDF text extraction
  - python-docx    : Word (.docx) text extraction
  - Built-in open(): plain-text (.txt / .md) reading

If pyresparser is not installed or fails for a particular file, the module
falls back gracefully to filename-based naming and empty structured fields.
"""

import os
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None  # type: ignore

# pyresparser — structured resume information extraction (project requirement)
# Ensure NLTK data and spaCy model are available before importing pyresparser.
try:
    import nltk as _nltk
    for _res in ("stopwords", "punkt", "averaged_perceptron_tagger", "punkt_tab"):
        _nltk.download(_res, quiet=True)
except Exception:  # pragma: no cover
    pass

try:
    import spacy as _spacy
    try:
        _spacy.load("en_core_web_sm")
    except OSError:
        # Model missing — download it automatically so pyresparser can run.
        print("[resume_parser] Downloading spaCy model 'en_core_web_sm' (first-run, one-time) ...")
        from spacy.cli import download as _spacy_dl  # noqa: PLC0415
        _spacy_dl("en_core_web_sm")
except Exception:  # pragma: no cover
    pass

try:
    from pyresparser import ResumeParser as PyResParser
    _HAS_PYRESPARSER = True
except Exception:  # pragma: no cover  — catches ImportError *and* NLTK LookupError
    _HAS_PYRESPARSER = False
    print("[resume_parser] pyresparser not available; using fallback extraction.")


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


def extract_structured_data(file_path: str) -> dict:
    """
    Extract structured information from a resume using pyresparser.

    pyresparser leverages spaCy NLP and NLTK to pull out:
      - name, email, mobile_number
      - skills (list)
      - education, degree
      - experience, total_experience

    Falls back to an empty-fields dict when pyresparser is unavailable
    or fails for a given file.

    Parameters
    ----------
    file_path : str
        Path to the resume file (PDF or DOCX work best with pyresparser).

    Returns
    -------
    dict
        Structured resume data.
    """
    if _HAS_PYRESPARSER:
        try:
            data = PyResParser(file_path).get_extracted_data()
            return {
                "name":             data.get("name", ""),
                "email":            data.get("email", ""),
                "mobile_number":    data.get("mobile_number", ""),
                "skills":           data.get("skills", []),
                "education":        data.get("education", []),
                "experience":       data.get("experience", []),
                "total_experience": data.get("total_experience", 0),
                "degree":           data.get("degree", []),
            }
        except Exception as exc:
            print(f"[resume_parser] pyresparser failed for {Path(file_path).name}: {exc}")

    # Fallback: minimal structured data derived from filename
    return {
        "name":             Path(file_path).stem,
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
            "name":       str,   # from pyresparser or filename stem
            "file":       str,   # full path
            "text":       str,   # raw extracted text
            "structured": dict,  # structured fields from pyresparser
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
            structured = extract_structured_data(str(file_path))
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
