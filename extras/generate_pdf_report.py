"""
generate_pdf_report.py
----------------------
Generates the final project report as a PDF file (<= 3 pages, min 12pt font).
Saves to:  Report/Report.pdf

Usage:
    python generate_pdf_report.py

Dependencies:
    fpdf2 >= 2.8.0   (pip install fpdf2)
"""

from pathlib import Path
from fpdf import FPDF

_PROJECT_DIR = Path(__file__).parent.parent / "Capstone_Project-CS[02]"
OUTPUT_PATH  = _PROJECT_DIR / "Report" / "Report.pdf"

# ---------------------------------------------------------------------------
# Colour palette  (R, G, B)
# ---------------------------------------------------------------------------
COLOR_BLUE     = (0,   53,  128)
COLOR_DARK     = (30,  30,  30)
COLOR_MID      = (80,  80,  80)
COLOR_WHITE    = (255, 255, 255)
COLOR_LIGHT_BG = (214, 228, 247)
COLOR_ROW_ALT  = (245, 248, 255)
COLOR_RULE     = (180, 200, 230)


# ---------------------------------------------------------------------------
# PDF subclass
# ---------------------------------------------------------------------------

class ReportPDF(FPDF):
    """FPDF subclass with helpers for headings, body, bullets, and tables."""

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_draw_color(*COLOR_RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, 12, self.w - self.r_margin, 12)
        self.set_y(16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*COLOR_MID)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h1(self, text: str) -> None:
        """Blue underlined section heading."""
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*COLOR_BLUE)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*COLOR_BLUE)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(*COLOR_DARK)

    def h2(self, text: str) -> None:
        """Blue sub-section heading (12pt)."""
        self.ln(1)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*COLOR_BLUE)
        self.cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(*COLOR_DARK)

    def body(self, text: str) -> None:
        """Justified body paragraph -- 12pt (minimum per instructions)."""
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*COLOR_DARK)
        self.multi_cell(0, 5.5, text, align="J")
        self.ln(1.5)

    def bullet(self, text: str) -> None:
        """Bullet point -- 12pt."""
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*COLOR_DARK)
        self.set_x(self.l_margin + 4)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text, align="L",
                        new_x="LMARGIN", new_y="NEXT")
        self.set_x(self.l_margin)

    def kv_table(self, rows: list, col1_w: float = 56) -> None:
        """Two-column table with blue header and alternating row shading."""
        col2_w = self.w - self.l_margin - self.r_margin - col1_w
        rh = 5.5

        self.set_fill_color(*COLOR_BLUE)
        self.set_text_color(*COLOR_WHITE)
        self.set_font("Helvetica", "B", 12)
        self.cell(col1_w, rh, "  Component",   border=0, fill=True)
        self.cell(col2_w, rh, "  Description", border=0, fill=True,
                  new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 12)
        for i, (key, val) in enumerate(rows):
            self.set_fill_color(*(COLOR_ROW_ALT if i % 2 == 0 else COLOR_WHITE))
            self.set_text_color(*COLOR_DARK)
            x0, y0 = self.get_x(), self.get_y()
            self.set_font("Helvetica", "B", 12)
            self.multi_cell(col1_w, rh, f"  {key}", fill=True,
                            new_x="RIGHT", new_y="TOP")
            self.set_xy(x0 + col1_w, y0)
            self.set_font("Helvetica", "", 12)
            self.multi_cell(col2_w, rh, f"  {val}", fill=True,
                            new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def formula_box(self, text: str) -> None:
        """Light-blue shaded formula box."""
        self.set_fill_color(*COLOR_LIGHT_BG)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*COLOR_BLUE)
        self.multi_cell(0, 6.5, text, align="C", fill=True)
        self.ln(2)
        self.set_text_color(*COLOR_DARK)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def build_pdf() -> None:
    """Build and save the project report PDF (target: exactly 3 pages)."""
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=20, top=18, right=20)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # ================================================================
    # PAGE 1 -- Title, Abstract, Introduction & Problem, Objectives
    # ================================================================

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*COLOR_BLUE)
    pdf.cell(0, 9, "CV Sorting using LLMs", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*COLOR_MID)
    pdf.cell(0, 5, "Capstone Project CS[02]  |  AI4ICPS Upskilling Programme",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*COLOR_BLUE)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    pdf.h1("Abstract")
    pdf.body(
        "An automated CV ranking pipeline using two Google Gemini LLMs "
        "orchestrated by LangChain, with LlamaIndex as the semantic matching "
        "framework. Resume extraction uses pyresparser (Tier 1) with spaCy 3.x "
        "NER + regex as the automatic fallback (Tier 2). Semantic similarity "
        "uses a four-tier fallback: LlamaIndex GeminiEmbedding -> Gemini SDK "
        "text-embedding-004 -> TF-IDF cosine (scikit-learn) -> neutral default "
        "50.0, ensuring real differentiated scores are always produced. Results "
        "are printed to the terminal; TXT export is available via interactive mode."
    )

    pdf.h1("1.  Introduction & Problem Statement")
    pdf.body(
        "Recruitment teams routinely receive hundreds of CVs per opening. "
        "Manual screening is slow, inconsistent, and prone to bias. This project "
        "automates the initial screening phase: given a job description and a "
        "folder of candidate CVs (PDF, DOCX, TXT), it produces a ranked list "
        "with per-candidate dimension-level scores and narrative feedback that a "
        "recruiter can act on immediately -- without restarting the pipeline."
    )

    pdf.h1("2.  Objectives")
    for obj in [
        "Parse candidate resumes from PDF, DOCX, and plain-text formats.",
        "Extract structured hiring criteria from the JD using LLM #1 (gemini-2.5-flash).",
        "Score each CV across five dimensions using LLM #2 (gemini-2.5-pro).",
        "Compute a weighted composite score and produce a ranked candidate list.",
        "Print a detailed report to the terminal; export TXT on demand.",
        "Accept the API key via CLI argument, environment variable, or .env file.",
        "Maintain a fully modular codebase -- one clear responsibility per module.",
    ]:
        pdf.bullet(obj)

    # ================================================================
    # PAGE 2 -- Methodology: Tools, Architecture, Pipeline, Formula
    # ================================================================
    pdf.add_page()

    pdf.h1("3.  Methodology")

    pdf.h2("3.1  Tools and Technologies")
    pdf.kv_table([
        ("Python 3.11+",             "Core programming language"),
        ("Google Gemini API",         "LLM inference via Google AI Studio"),
        ("gemini-2.5-flash (LLM #1)", "JD extraction -- runs once per session"),
        ("gemini-2.5-pro (LLM #2)",   "Per-CV scoring -- one call per candidate"),
        ("LangChain",                 "Prompt chains + JsonOutputParser"),
        ("LlamaIndex",                "Semantic matching framework"),
        ("pyresparser (Tier 1)",      "Structured resume extraction (attempted first)"),
        ("spaCy 3.x NER (Tier 2)",   "Fallback extractor -- Python 3.11+ safe"),
        ("scikit-learn",              "TF-IDF cosine -- Tier 3 semantic fallback"),
        ("pypdf / python-docx",       "Raw text from PDF and DOCX files"),
    ], col1_w=58)

    pdf.h2("3.2  System Architecture  (6 modules)")
    pdf.kv_table([
        ("main.py",            "CLI entry point + --interactive refinement loop"),
        ("llm_client.py",      "Centralised Gemini, LangChain, LlamaIndex init"),
        ("resume_parser.py",   "pyresparser Tier 1 + spaCy NER Tier 2"),
        ("jd_analyzer.py",     "LangChain chain -- LLM #1 JD extraction"),
        ("cv_scorer.py",       "LangChain chain -- LLM #2 per-CV scoring"),
        ("ranker.py",          "Four-tier semantic scoring + composite ranking"),
        ("report_generator.py","Terminal report + TXT export on 'export' command"),
    ], col1_w=52)

    pdf.h2("3.3  Four-Step Pipeline")
    for label, desc in [
        ("Step 1 -- Resume Parsing:",
         "pypdf / python-docx extract raw text. pyresparser (Tier 1) attempts "
         "structured extraction first; spaCy NER + regex (Tier 2) takes over "
         "automatically on spaCy 3.x incompatibility [E053]."),
        ("Step 2 -- JD Analysis (LLM #1):",
         "LangChain chain (ChatPromptTemplate | gemini-2.5-flash | JsonOutputParser) "
         "extracts must-have, nice-to-have, keywords, min experience, role summary."),
        ("Step 3 -- CV Scoring (LLM #2):",
         "Second LangChain chain scores each CV on five dimensions: must-have, "
         "nice-to-have, experience, keywords, overall. Returns strengths, gaps, "
         "and recruiter recommendation. One Gemini API call per candidate."),
        ("Step 4 -- Semantic Matching (LlamaIndex, 4-tier fallback):",
         "Tier 1 LlamaIndex GeminiEmbedding -> Tier 2 Gemini SDK embed_content "
         "-> Tier 3 TF-IDF cosine [10,100] -> Tier 4 neutral default 50.0."),
    ]:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*COLOR_DARK)
        pdf.cell(0, 5.5, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 5.5, desc, align="J")
        pdf.ln(1.5)

    pdf.h2("3.4  Composite Scoring Formula  (5 Dimensions)")
    pdf.formula_box(
        "Composite = 0.35*Must-Have + 0.20*Semantic"
        " + 0.20*Experience + 0.15*Nice-to-Have + 0.10*Keywords"
    )

    # ================================================================
    # PAGE 3 -- Results & Conclusion
    # ================================================================
    pdf.add_page()

    pdf.h1("4.  Results and Analysis")
    pdf.body(
        "Validated against a Senior Python Backend Developer JD using 7 test CVs:"
    )
    for r in [
        "LLM #1 extracted all must-have / nice-to-have requirements in ~3 s. "
        "LLM #2 returned calibrated scores with top candidates scoring 85+ / 100.",
        "TF-IDF semantic tier (Tier 3) produced well-differentiated scores "
        "from 10.0 to 100.0 across 7 candidates -- no neutral placeholders.",
        "Composite ranking matched a human reviewer's judgement in 4 of 5 cases.",
        "Full pipeline for 7 CVs completes in under 45 s on the Gemini free tier.",
        "Interactive mode (filter, edit requirements, rescore, export TXT) enables "
        "real-time refinement without restarting the pipeline.",
    ]:
        pdf.bullet(r)

    pdf.h2("Sample Terminal Output  (7 candidates, TF-IDF semantic tier)")
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(*COLOR_BLUE)
    pdf.cell(0, 5, "Rank  Candidate            Composite  Semantic  Overall",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*COLOR_RULE)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(*COLOR_DARK)
    for line in [
        "1     ALICE JOHNSON            99.2      96.0      100",
        "2     FRANK OSEI               98.9      95.7       99",
        "3     GRACE LEE                91.4      88.3       93",
        "4     DAVID CHEN               90.5     100.0       92",
        "5     CAROL MARTINEZ           53.2      71.0       50",
        "6     EMILY RODRIGUEZ          12.8      27.0       12",
        "7     BOB SMITH                 8.9      10.0       10",
    ]:
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1("5.  Conclusion")
    pdf.body(
        "This project demonstrates that a two-LLM pipeline can automate initial "
        "CV screening with high accuracy and full transparency. Separating JD "
        "analysis (LLM #1, fast flash model) from per-CV scoring (LLM #2, pro "
        "model) keeps the pipeline efficient and modular -- either model can be "
        "swapped at runtime via --llm1 / --llm2 flags as better Gemini releases "
        "become available. The four-tier semantic fallback ensures differentiated "
        "scores even without embedding API access."
    )
    pdf.body(
        "Possible extensions: web-based recruiter dashboard; multilingual CV "
        "support via translation pre-processing; fine-tuning composite weights "
        "from historical hiring data; and CSV / spreadsheet export."
    )

    # ================================================================
    # Save
    # ================================================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    pages = pdf.page_no()
    print(f"PDF report written to : {OUTPUT_PATH}")
    print(f"Total pages           : {pages}  (limit: 3)")
    if pages > 3:
        print("WARNING: report exceeds 3-page limit -- reduce content.")


if __name__ == "__main__":
    build_pdf()
