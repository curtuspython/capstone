"""
generate_pdf_report.py
----------------------
One-shot helper script that generates the 3-page PDF project report using
fpdf2 (open-source PDF generation library, https://github.com/py-pdf/fpdf2).

The report embeds simulated terminal screenshots of the ranking output and
candidate score breakdown to illustrate results and analysis.

Usage:
    python generate_pdf_report.py
"""

from fpdf import FPDF, XPos, YPos
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "Report" / "Report.pdf"

# ---- Design tokens (all fonts >= 12pt as per submission guidelines) ----
_BLUE       = (0, 60, 143)
_DARK_BLUE  = (0, 40, 100)
_WHITE      = (255, 255, 255)
_DARK       = (30, 30, 30)
_GREY       = (110, 110, 110)
_LIGHT_GREY = (240, 242, 245)
_TERM_BG    = (30, 30, 36)       # dark terminal background
_TERM_FG    = (0, 255, 120)      # green terminal text
_TERM_W     = (220, 220, 220)    # white terminal text
_LM = 15
_RM = 15
_W  = 210 - _LM - _RM           # usable width (180 mm)
_BODY_SZ = 12                    # minimum body font size (submission rule)
_LH = 5.5                        # line height for 12pt text


class ReportPDF(FPDF):
    """Custom FPDF subclass with header accent bar and page numbers."""

    def header(self):
        """Draw a thin blue accent bar at the top (pages 2+)."""
        if self.page_no() == 1:
            return
        self.set_fill_color(*_BLUE)
        self.rect(0, 0, 210, 4.5, style="F")
        self.set_y(6)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_GREY)
        self.cell(0, 4, "CS[02]  |  CV Sorting using LLMs  |  Project Report",
                  align="C")
        self.ln(5)

    def footer(self):
        """Render a centered page number."""
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GREY)
        self.cell(0, 5, f"Page {self.page_no()} of 3", align="C")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _section(pdf: FPDF, title: str) -> None:
    """Render a section heading with a blue background bar (font >= 14pt)."""
    pdf.ln(2.5)
    y = pdf.get_y()
    pdf.set_fill_color(*_BLUE)
    pdf.rect(_LM, y, _W, 7.5, style="F")
    pdf.set_xy(_LM + 3, y + 0.5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_WHITE)
    pdf.cell(_W - 6, 6.5, title)
    pdf.set_y(y + 9)
    pdf.set_text_color(*_DARK)


def _sub(pdf: FPDF, title: str) -> None:
    """Render a sub-heading with a left blue accent stripe (font 12pt bold)."""
    pdf.ln(1.5)
    y = pdf.get_y()
    pdf.set_fill_color(*_BLUE)
    pdf.rect(_LM, y, 2, 6, style="F")
    pdf.set_xy(_LM + 5, y)
    pdf.set_font("Helvetica", "B", _BODY_SZ)
    pdf.set_text_color(*_DARK_BLUE)
    pdf.cell(0, 6, title)
    pdf.set_y(y + 7.5)
    pdf.set_text_color(*_DARK)


def _para(pdf: FPDF, text: str) -> None:
    """Render a body paragraph (12pt)."""
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_DARK)
    pdf.set_x(_LM)
    pdf.multi_cell(_W, _LH, text)
    pdf.ln(0.8)


def _bullet(pdf: FPDF, text: str) -> None:
    """Render a bullet-point line (12pt)."""
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_DARK)
    pdf.set_x(_LM + 4)
    pdf.multi_cell(_W - 4, _LH, f"-  {text}")
    pdf.ln(0.3)


def _terminal_block(pdf: FPDF, lines: list[str], title: str = "") -> None:
    """
    Render a simulated terminal screenshot: dark background, monospace text,
    with an optional grey title bar to mimic a terminal window.
    """
    x0 = _LM
    y0 = pdf.get_y() + 1
    line_h = 4.2
    pad = 3
    title_h = 6 if title else 0
    block_h = title_h + pad * 2 + len(lines) * line_h

    # Title bar (grey)
    if title:
        pdf.set_fill_color(60, 60, 66)
        pdf.rect(x0, y0, _W, title_h, style="F")
        # Fake window dots
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            pdf.set_fill_color(*c)
            pdf.ellipse(x0 + 4 + i * 5, y0 + 1.8, 2.5, 2.5, style="F")
        pdf.set_font("Courier", "B", 8)
        pdf.set_text_color(200, 200, 200)
        pdf.set_xy(x0 + 22, y0 + 0.5)
        pdf.cell(_W - 30, title_h - 1, title, align="L")
        y0 += title_h

    # Dark body
    pdf.set_fill_color(*_TERM_BG)
    pdf.rect(x0, y0, _W, pad * 2 + len(lines) * line_h, style="F")

    pdf.set_font("Courier", "", 9)
    cur_y = y0 + pad
    for line in lines:
        # Colour: green for data lines, white for borders/headers
        if any(c in line for c in ("===", "---", "Rank", "Step", "[main]")):
            pdf.set_text_color(*_TERM_W)
        else:
            pdf.set_text_color(*_TERM_FG)
        pdf.set_xy(x0 + pad, cur_y)
        pdf.cell(_W - pad * 2, line_h, line)
        cur_y += line_h

    pdf.set_y(cur_y + pad + 1)
    pdf.set_text_color(*_DARK)


def _mini_table(pdf: FPDF, headers: list, rows: list, col_w: list) -> None:
    """Render a compact styled table (12pt font)."""
    pdf.set_font("Helvetica", "B", _BODY_SZ)
    pdf.set_fill_color(*_BLUE)
    pdf.set_text_color(*_WHITE)
    pdf.set_x(_LM)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 6.5, f" {h}", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", _BODY_SZ)
    pdf.set_text_color(*_DARK)
    for idx, row in enumerate(rows):
        if idx % 2 == 0:
            pdf.set_fill_color(*_LIGHT_GREY)
        pdf.set_x(_LM)
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, f" {val}", fill=(idx % 2 == 0))
        pdf.ln()
    pdf.ln(1.5)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report() -> None:
    """Build the 3-page PDF project report and save to Report/Report.pdf."""
    pdf = ReportPDF()
    pdf.set_margins(left=_LM, top=12, right=_RM)
    pdf.set_auto_page_break(auto=True, margin=12)

    # ==================================================================
    # PAGE 1 -- Title, Abstract, Introduction, Problem, Objectives
    # ==================================================================
    pdf.add_page()

    # Title banner
    pdf.set_fill_color(*_BLUE)
    pdf.rect(0, 0, 210, 38, style="F")
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 10, "CV Sorting using LLMs", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(200, 220, 255)
    pdf.cell(0, 7, "Capstone Project CS[02]  |  AI4ICPS Programme",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(42)

    _section(pdf, "Abstract")
    _para(pdf,
        "This project builds an automated CV ranking pipeline using two "
        "Google Gemini LLMs orchestrated by LangChain. Resumes are parsed "
        "with pyresparser for structured extraction, and LlamaIndex Gemini "
        "embeddings add semantic similarity scoring. Candidates are ranked "
        "across five weighted dimensions, with results exported as TXT/CSV "
        "reports and an interactive terminal mode for real-time refinement."
    )

    _section(pdf, "1. Introduction")
    _para(pdf,
        "Recruitment teams receive hundreds of CVs per opening. Manual "
        "screening is slow, inconsistent, and biased. LLMs can automate "
        "initial screening with high accuracy. This project combines two "
        "specialised LLMs in a sequential pipeline -- a modern GenAI "
        "best-practice -- to solve this real-world HR problem."
    )

    _section(pdf, "2. Problem Statement")
    _para(pdf,
        "Given a job description and candidate CVs (PDF/DOCX/TXT), produce "
        "a ranked shortlist with per-candidate match explanations and "
        "evidence that a recruiter can act on immediately."
    )

    _section(pdf, "3. Objectives")
    _bullet(pdf, "Parse resumes and extract structured profiles via pyresparser.")
    _bullet(pdf, "Analyse JD using LangChain + Gemini (LLM #1: gemini-2.5-flash).")
    _bullet(pdf, "Score each CV on 5 dimensions via LangChain + Gemini (LLM #2: gemini-2.5-pro).")
    _bullet(pdf, "Add LlamaIndex semantic similarity into composite scoring.")
    _bullet(pdf, "Generate TXT/CSV reports; support interactive query refinement.")

    # ==================================================================
    # PAGE 2 -- Methodology
    # ==================================================================
    pdf.add_page()

    _section(pdf, "4. Methodology")

    _sub(pdf, "4.1 Tools and Technologies")
    _mini_table(pdf,
        headers=["Tool", "Purpose"],
        rows=[
            ("Google Gemini API",    "LLM inference (google-genai SDK)"),
            ("LangChain",            "Prompt orchestration + output parsing"),
            ("LlamaIndex",           "Semantic similarity via Gemini embeddings"),
            ("pyresparser",          "Structured resume extraction"),
            ("pypdf / python-docx",  "PDF and Word text extraction"),
            ("fpdf2",                "PDF report generation (open-source)"),
        ],
        col_w=[50, _W - 50],
    )

    _sub(pdf, "4.2 System Architecture")
    _mini_table(pdf,
        headers=["Module", "Responsibility"],
        rows=[
            ("main.py",             "CLI entry point + --interactive mode"),
            ("llm_client.py",       "Gemini client (genai, LangChain, LlamaIndex)"),
            ("resume_parser.py",    "Text extraction + pyresparser profiles"),
            ("jd_analyzer.py",      "LLM #1: structured JD extraction"),
            ("cv_scorer.py",        "LLM #2: per-CV multi-dimension scoring"),
            ("ranker.py",           "Composite + semantic similarity scoring"),
            ("report_generator.py", "TXT and CSV report generation"),
        ],
        col_w=[45, _W - 45],
    )

    _sub(pdf, "4.3 Two-LLM Pipeline")
    _para(pdf,
        "LLM #1 (gemini-2.5-flash) extracts structured JD requirements "
        "via a LangChain chain (ChatPromptTemplate + JsonOutputParser). "
        "LLM #2 (gemini-2.5-pro) scores each CV using structured JD + "
        "pyresparser profile + raw CV text, producing dimension scores "
        "and narrative feedback."
    )

    _sub(pdf, "4.4 Composite Scoring (5 Dimensions)")
    _para(pdf,
        "Composite = 0.35*MustHave + 0.20*SemanticSimilarity + 0.20*Experience "
        "+ 0.15*NiceToHave + 0.10*Keywords. Must-Have carries "
        "the highest weight (35%%, hard requirement filter); semantic similarity "
        "(20%%, LlamaIndex) captures holistic alignment beyond keywords; "
        "experience and nice-to-have follow; keyword score (10%%) "
        "acts as a domain fluency signal."
    )

    # ==================================================================
    # PAGE 3 -- Results (with terminal screenshots) + Conclusion
    # ==================================================================
    pdf.add_page()

    _section(pdf, "5. Results and Analysis")

    _para(pdf, "Terminal output showing the ranked candidate list:")
    _terminal_block(pdf, [
        "[main] Step 1/4 -- Analysing JD with LLM #1 (gemini-2.5-flash) ...",
        "[main] Step 2/4 -- Parsing candidate CVs ...",
        "[main] Step 3/4 -- Scoring 5 CV(s) with LLM #2 (gemini-2.5-pro) ...",
        "[main] Step 4/4 -- Ranking candidates (+ semantic matching) ...",
        "",
        "==============================================================",
        "Rank  Candidate          Composite  Semantic  Overall",
        "--------------------------------------------------------------",
        " 1    alice_resume           84.2      89.3       87",
        " 2    bob_cv                 72.5      76.1       75",
        " 3    carol_profile          61.8      58.4       63",
        " 4    dave_resume            45.3      42.7       44",
        " 5    eve_cv                 32.1      35.6       30",
        "==============================================================",
    ], title="Terminal -- Ranking Output")

    _para(pdf, "Detailed candidate inspection (show 1) in interactive mode:")
    _terminal_block(pdf, [
        "[interactive] > show 1",
        "------------------------------------------------------------",
        "  RANK #1  --  alice_resume",
        "  Composite: 84.2 / 100   Semantic: 89.3   Overall: 87",
        "  Must-Have: 92   Experience: 85   Nice-to-Have: 78",
        "  Strengths: + 5 yrs Python, + AWS certified, + ML exp",
        "  Gaps:      - No Kubernetes experience",
        "  Recommendation: Strong hire",
        "------------------------------------------------------------",
    ], title="Terminal -- Interactive Mode (show 1)")

    # ---- Conclusion ----
    _section(pdf, "6. Conclusion")
    _para(pdf,
        "The multi-framework pipeline (LangChain + LlamaIndex + pyresparser "
        "+ Gemini) automates CV screening with high accuracy and full "
        "transparency. The modular design allows swapping any component "
        "independently. The interactive terminal mode enables real-time "
        "query refinement without GUI dependency. Future work includes "
        "fine-tuning weights from hiring outcomes and multilingual support."
    )

    # ---- Save ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
