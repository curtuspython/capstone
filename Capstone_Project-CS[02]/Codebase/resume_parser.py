"""
resume_parser.py
----------------
Parses candidate resumes from PDF, DOCX, or plain-text files and returns
a structured dictionary containing the candidate's name and raw text.

Supported formats:
  - PDF   (.pdf)  via pypdf
  - Word  (.docx) via python-docx
  - Text  (.txt)  via built-in open()
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


def parse_resumes_from_directory(directory: str) -> list[dict]:
    """
    Scan a directory for resume files and return a list of candidate dicts.

    Each dict has the shape::

        {
            "name": str,   # derived from filename (stem)
            "file": str,   # full path
            "text": str,   # raw extracted text
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
            candidates.append({
                "name": file_path.stem,   # filename without extension
                "file": str(file_path),
                "text": text.strip(),
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
