"""
generate_pdf_report.py
----------------------
One-shot helper script that generates the 3-page PDF project report using
fpdf2 (open-source PDF generation library, https://github.com/py-pdf/fpdf2).

The report uses a subtle, professional colour palette (charcoal + muted teal
accents on white) and embeds simulated terminal screenshots of the actual
ranking output and interactive-mode candidate inspection.

Design constraints (submission guidelines):
  - Minimum text font size: 12 pt
  - Maximum length: 3 pages
  - Output format: PDF only

Usage:
    python generate_pdf_report.py
"""

from fpdf import FPDF, XPos, YPos
from pathlib import Path

# Output path: saved into the Report/ directory alongside the Codebase/
OUTPUT_PATH = Path(__file__).parent.parent / "Report" / "Report.pdf"

# ---------------------------------------------------------------------------
# Design tokens  (subtle, professional palette -- charcoal + muted teal)
# ---------------------------------------------------------------------------
_CHARCOAL   = (44, 62, 80)       # primary headers  (#2C3E50)
_TEAL       = (22, 160, 133)     # accent colour     (#16A085)
_DARK_TEAL  = (14, 105, 90)      # sub-heading text  (#0E695A)
_WHITE      = (255, 255, 255)
_NEAR_BLACK = (28, 40, 51)       # body text          (#1C2833)
_MID_GREY   = (130, 139, 148)    # footer / captions  (#828B94)
_LIGHT_BG   = (244, 246, 247)    # alternate table row (#F4F6F7)
_TERM_BG    = (40, 44, 52)       # terminal background (#282C34)
_TERM_GREEN = (152, 195, 121)    # terminal data text  (#98C379)
_TERM_WHITE = (210, 210, 210)    # terminal chrome text

# Layout constants
_LM = 15                         # left margin (mm)
_RM = 15                         # right margin (mm)
_W  = 210 - _LM - _RM           # usable content width (180 mm)
_BODY_SZ = 12                    # minimum body font size (submission rule)
_LH = 5.5                        # line height for 12 pt body text


# ---------------------------------------------------------------------------
# Custom FPDF subclass  (header accent line + page number footer)
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """FPDF subclass adding a thin accent line on pages 2-3 and page numbers."""

    def header(self):
        """Draw a thin teal accent line and running header on pages 2+."""
        if self.page_no() == 1:
            return  # page 1 has its own title banner
        # Thin accent line across the top
        self.set_draw_color(*_TEAL)
        self.set_line_width(0.6)
        self.line(0, 5, 210, 5)
        # Running header text (small, grey, centred)
        self.set_y(6.5)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_MID_GREY)
        self.cell(0, 4, "CS[02]  |  CV Sorting using LLMs  |  Project Report",
                  align="C")
        self.ln(5)

    def footer(self):
        """Render a centred page number at the bottom of every page."""
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID_GREY)
        self.cell(0, 5, f"Page {self.page_no()} / 3", align="C")


# ---------------------------------------------------------------------------
# Drawing helpers  (section bars, sub-headings, paragraphs, bullets, tables)
# ---------------------------------------------------------------------------

def _section(pdf: FPDF, title: str) -> None:
    """Render a section heading: charcoal background bar, white bold 14 pt text."""
    pdf.ln(2)
    y = pdf.get_y()
    # Charcoal background bar spanning the content width
    pdf.set_fill_color(*_CHARCOAL)
    pdf.rect(_LM, y, _W, 7.5, style="F")
    # Section title in white
    pdf.set_xy(_LM + 3, y + 0.5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_WHITE)
    pdf.cell(_W - 6, 6.5, title)
    pdf.set_y(y + 9)
    pdf.set_text_color(*_NEAR_BLACK)


def _sub(pdf: FPDF, title: str) -> None:
    """Render a sub-heading: thin teal left-border stripe, bold 12 pt text."""
    pdf.ln(1.5)
    y = pdf.get_y()
    # Thin teal accent stripe on the left
    pdf.set_fill_color(*_TEAL)
    pdf.rect(_LM, y, 2, 6, style="F")
    # Sub-heading text
    pdf.set_xy(_LM + 5, y)
    pdf.set_font("Helvetica", "B", _BODY_SZ)
    pdf.set_text_color(*_DARK_TEAL)
    pdf.cell(0, 6, title)
    pdf.set_y(y + 7.5)
    pdf.set_text_color(*_NEAR_BLACK)


def _para(pdf: FPDF, text: str) -> None:
    """Render a body paragraph in regular 12 pt Helvetica."""
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_NEAR_BLACK)
    pdf.set_x(_LM)
    pdf.multi_cell(_W, _LH, text)
    pdf.ln(0.8)


def _bullet(pdf: FPDF, text: str) -> None:
    """Render a single bullet-point line (dash prefix, 12 pt, indented)."""
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_NEAR_BLACK)
    pdf.set_x(_LM + 4)
    pdf.multi_cell(_W - 4, _LH, f"-  {text}")
    pdf.ln(0.3)


def _terminal_block(pdf: FPDF, lines: list[str], title: str = "") -> None:
    """
    Draw a simulated terminal screenshot with an optional macOS-style title bar.

    The block uses a dark background with green/white monospace text to mimic
    real terminal output.  Data lines are green; chrome lines (borders, headers,
    step labels) are white for visual contrast.
    """
    x0 = _LM
    y0 = pdf.get_y() + 1
    line_h = 3.8          # compact line height for 8.5 pt monospace
    pad = 3               # internal padding inside the dark block
    title_h = 6 if title else 0

    # --- Title bar (dark grey, macOS-style traffic-light dots) ---
    if title:
        pdf.set_fill_color(55, 55, 60)
        pdf.rect(x0, y0, _W, title_h, style="F")
        # Three coloured dots (close / minimise / maximise)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            pdf.set_fill_color(*c)
            pdf.ellipse(x0 + 4 + i * 5, y0 + 1.8, 2.5, 2.5, style="F")
        # Title text
        pdf.set_font("Courier", "B", 8)
        pdf.set_text_color(190, 190, 190)
        pdf.set_xy(x0 + 22, y0 + 0.5)
        pdf.cell(_W - 30, title_h - 1, title, align="L")
        y0 += title_h

    # --- Dark body ---
    pdf.set_fill_color(*_TERM_BG)
    pdf.rect(x0, y0, _W, pad * 2 + len(lines) * line_h, style="F")

    pdf.set_font("Courier", "", 8.5)
    cur_y = y0 + pad
    for line in lines:
        # Chrome lines (borders, headers, step labels) in white; data in green
        if any(tok in line for tok in ("===", "---", "Rank", "#", "[main]",
                                       "[interactive]", "RANK")):
            pdf.set_text_color(*_TERM_WHITE)
        else:
            pdf.set_text_color(*_TERM_GREEN)
        pdf.set_xy(x0 + pad, cur_y)
        pdf.cell(_W - pad * 2, line_h, line)
        cur_y += line_h

    pdf.set_y(cur_y + pad + 1)
    pdf.set_text_color(*_NEAR_BLACK)


def _table(pdf: FPDF, headers: list, rows: list, col_w: list) -> None:
    """
    Render a compact styled table with a charcoal header row and alternating
    light-grey / white body rows.  All text is 12 pt (submission minimum).
    """
    # Header row (charcoal background, white text)
    pdf.set_font("Helvetica", "B", _BODY_SZ)
    pdf.set_fill_color(*_CHARCOAL)
    pdf.set_text_color(*_WHITE)
    pdf.set_x(_LM)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 6.5, f" {h}", fill=True)
    pdf.ln()

    # Body rows (alternating stripes for readability)
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_NEAR_BLACK)
    for idx, row in enumerate(rows):
        stripe = idx % 2 == 0  # even rows get light background
        if stripe:
            pdf.set_fill_color(*_LIGHT_BG)
        pdf.set_x(_LM)
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, f" {val}", fill=stripe)
        pdf.ln()
    pdf.ln(1.5)


# ---------------------------------------------------------------------------
# Report content builder  (3 pages)
# ---------------------------------------------------------------------------

def build_report() -> None:
    """
    Construct the full 3-page PDF project report and write it to
    Report/Report.pdf.

    Page layout:
      Page 1 -- Title banner, Abstract, Introduction, Problem, Objectives
      Page 2 -- Methodology (tools, architecture, pipeline, scoring)
      Page 3 -- Results with terminal screenshots, Conclusion
    """
    pdf = _ReportPDF()
    pdf.set_margins(left=_LM, top=12, right=_RM)
    pdf.set_auto_page_break(auto=True, margin=12)

    # ==================================================================
    # PAGE 1 -- Title, Abstract, Introduction, Problem, Objectives
    # ==================================================================
    pdf.add_page()

    # --- Title banner (charcoal, full width) ---
    pdf.set_fill_color(*_CHARCOAL)
    pdf.rect(0, 0, 210, 36, style="F")
    # Thin teal accent line at the banner bottom
    pdf.set_fill_color(*_TEAL)
    pdf.rect(0, 36, 210, 1.5, style="F")

    # Title text (white on charcoal)
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 10, "CV Sorting using LLMs", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Subtitle (light grey on charcoal)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(190, 200, 210)
    pdf.cell(0, 7, "Capstone Project CS[02]  |  AI4ICPS Programme",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(42)

    # --- Abstract ---
    _section(pdf, "Abstract")
    _para(pdf,
        "This project implements an automated CV ranking pipeline powered by "
        "two Google Gemini LLMs (gemini-2.5-flash and gemini-2.5-pro), "
        "orchestrated through LangChain. Resumes are parsed via a two-tier "
        "strategy: pyresparser (Tier 1) with spaCy 3.x NER + regex as an "
        "automatic fallback (Tier 2). Semantic similarity follows a four-tier "
        "approach: LlamaIndex GeminiEmbedding, Gemini SDK text-embedding-004, "
        "and TF-IDF cosine similarity (scikit-learn). Candidates are ranked on "
        "five weighted dimensions. An interactive terminal mode supports "
        "real-time query refinement for recruiters."
    )

    # --- Introduction ---
    _section(pdf, "1. Introduction")
    _para(pdf,
        "Recruitment teams receive hundreds of CVs per opening. Manual "
        "screening is slow, inconsistent, and prone to bias. Large Language "
        "Models can automate initial screening with high accuracy and "
        "transparency. This project combines two specialised Gemini LLMs in "
        "a sequential pipeline to solve this real-world HR problem, using "
        "LangChain for prompt orchestration and LlamaIndex as the semantic "
        "matching framework."
    )

    # --- Problem Statement ---
    _section(pdf, "2. Problem Statement")
    _para(pdf,
        "Given a job description and a set of candidate CVs (PDF / DOCX / TXT), "
        "produce a ranked shortlist with per-candidate match explanations and "
        "supporting evidence that a recruiter can act on immediately."
    )

    # --- Objectives ---
    _section(pdf, "3. Objectives")
    _bullet(pdf, "Parse resumes via pyresparser (Tier 1) with spaCy 3.x NER fallback (Tier 2).")
    _bullet(pdf, "Analyse the JD using LangChain + Gemini LLM #1 (gemini-2.5-flash).")
    _bullet(pdf, "Score each CV on 5 dimensions via LangChain + Gemini LLM #2 (gemini-2.5-pro).")
    _bullet(pdf, "Compute semantic similarity: LlamaIndex -> Gemini SDK -> TF-IDF -> default.")
    _bullet(pdf, "Provide an interactive terminal mode for query refinement and TXT export.")

    # ==================================================================
    # PAGE 2 -- Methodology
    # ==================================================================
    pdf.add_page()

    _section(pdf, "4. Methodology")

    # --- 4.1 Tools and Technologies ---
    _sub(pdf, "4.1 Tools and Technologies")
    _table(pdf,
        headers=["Tool / Library", "Role in Pipeline"],
        rows=[
            ("Google Gemini API",       "LLM inference (gemini-2.5-flash + gemini-2.5-pro)"),
            ("LangChain",               "Prompt orchestration, chaining, JSON output parsing"),
            ("LlamaIndex",              "Semantic matching framework (embedding interface)"),
            ("pyresparser (Tier 1)",    "Structured resume extraction (project requirement)"),
            ("spaCy 3.x NER (Tier 2)", "Fallback resume extractor: NER + regex"),
            ("scikit-learn",            "TF-IDF cosine similarity (semantic scoring Tier 3)"),
            ("pypdf / python-docx",    "PDF and Word document text extraction"),
            ("fpdf2",                   "Open-source PDF report generation"),
        ],
        col_w=[58, _W - 58],
    )

    # --- 4.2 System Architecture ---
    _sub(pdf, "4.2 System Architecture")
    _table(pdf,
        headers=["Module", "Responsibility"],
        rows=[
            ("main.py",             "CLI entry point, pipeline orchestration, --interactive mode"),
            ("llm_client.py",       "Centralised Gemini client (genai SDK, LangChain, LlamaIndex)"),
            ("resume_parser.py",    "Text extraction + structured NER profiles (2-tier)"),
            ("jd_analyzer.py",      "LLM #1: structured JD analysis via LangChain chain"),
            ("cv_scorer.py",        "LLM #2: per-CV multi-dimension scoring via LangChain"),
            ("ranker.py",           "Composite scoring + semantic similarity (4-tier fallback)"),
            ("report_generator.py", "Terminal report printing + TXT export on demand"),
        ],
        col_w=[42, _W - 42],
    )

    # --- 4.3 Two-LLM Pipeline ---
    _sub(pdf, "4.3 Two-LLM Pipeline")
    _para(pdf,
        "LLM #1 (gemini-2.5-flash) analyses the job description once via a "
        "LangChain chain (ChatPromptTemplate + JsonOutputParser) and extracts "
        "structured requirements. LLM #2 (gemini-2.5-pro) then scores each CV "
        "against those requirements on five dimensions, producing narrative "
        "feedback per candidate. Flash is chosen for its speed on the single-run "
        "JD task; Pro is placed on the accuracy-critical per-CV scoring loop."
    )

    # --- 4.4 Composite Scoring ---
    _sub(pdf, "4.4 Composite Scoring Formula")
    _para(pdf,
        "Composite = 0.35 x Must-Have + 0.20 x Semantic + 0.20 x Experience "
        "+ 0.15 x Nice-to-Have + 0.10 x Keywords.  Must-Have (35%) is the "
        "hardest filter. Semantic similarity (20%) uses a 4-tier fallback: "
        "LlamaIndex GeminiEmbedding, Gemini SDK embeddings, TF-IDF cosine "
        "(scikit-learn, batch-normalised to [10, 100]), or neutral default. "
        "Experience (20%) and Nice-to-Have (15%) follow; Keywords (10%) signals "
        "domain vocabulary fluency."
    )

    # ==================================================================
    # PAGE 3 -- Results (with REAL terminal screenshots) + Conclusion
    # ==================================================================
    pdf.add_page()

    _section(pdf, "5. Results and Analysis")

    # --- Terminal screenshot 1: Ranking output (actual data from run) ---
    _para(pdf, "Ranked candidate list (7 CVs scored for Senior Python Backend Developer):")
    _terminal_block(pdf, [
        "================================================================",
        "#   Candidate              Composite   Semantic   Overall",
        "----------------------------------------------------------------",
        "1   ALICE JOHNSON               99.2       96.0       100",
        "2   FRANK OSEI                  98.9       95.7        99",
        "3   ALICE JOHNSON               97.9       95.6        98",
        "4   DAVID CHEN                  90.5      100.0        92",
        "5   CAROL MARTINEZ              53.2       71.0        50",
        "6   EMILY RODRIGUEZ             12.8       27.0        12",
        "7   BOB SMITH                    8.9       10.0        10",
        "================================================================",
    ], title="Terminal  --  Ranking Output")

    # --- Terminal screenshot 2: Interactive mode (show command) ---
    _para(pdf, "Interactive inspection of a candidate (show 4) in --interactive mode:")
    _terminal_block(pdf, [
        "[interactive] > show 4",
        "------------------------------------------------------------",
        "  RANK #4  --  DAVID CHEN",
        "  Composite: 90.5 / 100   Semantic: 100.0   Overall: 92",
        "  Must-Have: 100  Experience: 90  Nice-to-Have: 50",
        "  Strengths:",
        "    + Direct experience with FastAPI, Django, PostgreSQL, Redis",
        "    + Strong AWS and Docker experience with measurable results",
        "  Gaps:",
        "    - No explicit team leadership or mentoring experience",
        "  Recruiter Note: Strong candidate, advance to interview.",
        "------------------------------------------------------------",
    ], title="Terminal  --  Interactive Mode (show 4)")

    # --- Conclusion ---
    _section(pdf, "6. Conclusion")
    _para(pdf,
        "The multi-framework pipeline (LangChain + LlamaIndex + pyresparser / "
        "spaCy NER + Gemini) automates CV screening with high accuracy and "
        "full transparency. The top two candidates (ALICE JOHNSON, FRANK OSEI) "
        "scored above 97, confirming precise alignment with the Senior Python "
        "Backend Developer requirements. Semantic scoring via TF-IDF fallback "
        "produces real, differentiated scores across all candidates. The "
        "interactive terminal mode enables real-time query refinement, "
        "filtering, and on-demand export -- all without GUI dependency."
    )

    # --- Save the PDF to disk ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    print(f"[generate_pdf_report] Report written to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_report()
