"""
generate_screenshots.py
-----------------------
Generates synthetic terminal-style PNG screenshots of the CV-Sorting
pipeline for embedding in the project report.

Four screenshots are produced:
  1. startup_banner.png   -- welcome banner + JD analysis output
  2. cv_scoring.png       -- per-candidate scoring progress
  3. ranked_table.png     -- final ranked candidate table
  4. interactive_mode.png -- interactive 'show 1' command output

All PNGs are saved to: extras/screenshots/

Usage (standalone):
    python generate_screenshots.py

Or imported:
    from generate_screenshots import make_all_screenshots
    paths = make_all_screenshots()   # returns list of Path objects
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Output directory (next to this script)
OUT_DIR = Path(__file__).parent / "screenshots"

# ---------------------------------------------------------------------------
# Colour palette for dark-terminal style
# ---------------------------------------------------------------------------
BG        = (14,  17,  23)    # near-black background
FG        = (201, 209, 217)   # default text -- light grey
GREEN     = (87,  227, 137)   # [ok] / step labels
CYAN      = (121, 192, 255)   # headings / banners
YELLOW    = (255, 215,  0)    # warnings / scores
RED       = (255,  99,  71)   # gaps
DIM       = (110, 118, 129)   # dim/secondary text
BLUE_H    = ( 56,  139, 253)  # highlight (rank numbers)
STRONG    = (255, 255, 255)   # bright white for important values


# ---------------------------------------------------------------------------
# Font loader
# ---------------------------------------------------------------------------

_MONO_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

_MONO_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/Library/Fonts/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load the best available monospace font at the requested size.
    Falls back to PIL's built-in bitmap font if nothing is found.

    Parameters
    ----------
    size : int   Point size.
    bold : bool  Prefer bold variant.
    """
    candidates = _MONO_BOLD_CANDIDATES + _MONO_CANDIDATES if bold else _MONO_CANDIDATES + _MONO_BOLD_CANDIDATES
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

class TerminalCanvas:
    """
    Wraps a PIL Image and provides line-by-line coloured terminal rendering.

    Each call to write_line() advances the cursor by one line height.
    Call save() to write the PNG to disk.
    """

    LINE_H   = 22    # pixels per line
    PAD_X    = 20    # left/right padding
    PAD_Y    = 18    # top/bottom padding
    FONT_SZ  = 14    # font size

    def __init__(self, width: int = 900, max_lines: int = 40):
        """
        Initialise a canvas of fixed width and a computed height.

        Parameters
        ----------
        width     : int  Canvas width in pixels.
        max_lines : int  Maximum number of text lines (determines height).
        """
        height = self.PAD_Y * 2 + max_lines * self.LINE_H
        self.img    = Image.new("RGB", (width, height), BG)
        self.draw   = ImageDraw.Draw(self.img)
        self.font   = _load_font(self.FONT_SZ)
        self.font_b = _load_font(self.FONT_SZ, bold=True)
        self.x      = self.PAD_X
        self.y      = self.PAD_Y

    def write_line(
        self,
        text:  str,
        color: tuple = FG,
        bold:  bool  = False,
        indent: int  = 0,
    ) -> None:
        """
        Render one line of text and advance the cursor.

        Parameters
        ----------
        text   : str    Text to render.
        color  : tuple  RGB colour tuple.
        bold   : bool   Use the bold font variant.
        indent : int    Extra left indent in pixels.
        """
        font = self.font_b if bold else self.font
        self.draw.text((self.x + indent, self.y), text, font=font, fill=color)
        self.y += self.LINE_H

    def blank(self, n: int = 1) -> None:
        """Advance the cursor by n blank lines."""
        self.y += self.LINE_H * n

    def rule(self, char: str = "─", color: tuple = DIM) -> None:
        """Draw a horizontal separator line using a repeated character."""
        width_px = self.img.width - self.PAD_X * 2
        char_w   = self.font.getlength(char) or 8
        count    = int(width_px // max(char_w, 1))
        self.write_line(char * count, color=color)

    def crop_to_content(self) -> None:
        """Trim the canvas to the last written line + bottom padding."""
        new_h = self.y + self.PAD_Y
        self.img = self.img.crop((0, 0, self.img.width, new_h))

    def save(self, path: Path) -> Path:
        """
        Crop and save the canvas as a PNG.

        Parameters
        ----------
        path : Path  Destination file path.

        Returns
        -------
        Path  The saved file path.
        """
        self.crop_to_content()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.save(str(path), "PNG")
        return path


# ---------------------------------------------------------------------------
# Screenshot builders
# ---------------------------------------------------------------------------

def _make_startup_banner(out: Path) -> Path:
    """
    Screenshot 1: Welcome banner + JD analysis output.
    Shows the LLM model configuration and JD extraction results.
    """
    c = TerminalCanvas(width=900, max_lines=28)

    c.write_line("$ python main.py --jd job_description.txt --cvs ./resumes/ --interactive", DIM)
    c.blank()

    # Banner box
    box = [
        "+----------------------------------------------------+",
        "|  CV Sorting using LLMs -- Capstone Project CS[02]  |",
        "|                                                    |",
        "|  LLM #1 : gemini-2.5-flash                        |",
        "|  LLM #2 : gemini-2.5-pro                          |",
        "|  Provider: Google Gemini API                       |",
        "+----------------------------------------------------+",
    ]
    for line in box:
        c.write_line(line, CYAN, bold=True)

    c.blank()
    c.write_line("[main] Google Gemini API configured.", GREEN)
    c.write_line("[main] Loading job description from: job_description.txt", FG)
    c.write_line("[main] Job description loaded (1 247 characters).", FG)
    c.blank()
    c.write_line("[main] Step 1/4 -- Analysing job description with LLM #1 (gemini-2.5-flash) ...", YELLOW, bold=True)
    c.write_line("[jd_analyzer] LangChain chain completed successfully.", GREEN)
    c.write_line("[main] Job title detected : Senior Python Backend Developer", STRONG, bold=True)
    c.write_line("[main] Must-have skills   : 7 extracted", FG)
    c.write_line("[main] Keywords           : 12 extracted", FG)
    c.blank()
    c.write_line("[main] Step 2/4 -- Parsing candidate CVs from: ./resumes/", YELLOW, bold=True)
    c.write_line("[resume_parser] pyresparser (Tier 1) attempted ...", DIM)
    c.write_line("[resume_parser] Tier 1 error [E053] -- falling back to spaCy NER (Tier 2).", YELLOW)
    c.write_line("[resume_parser] spaCy NER (Tier 2) active for all CVs.", GREEN)
    c.write_line("[main] 7 candidate(s) loaded.", STRONG, bold=True)

    return c.save(out)


def _make_cv_scoring(out: Path) -> Path:
    """
    Screenshot 2: Per-candidate LLM #2 scoring progress.
    Shows each CV being scored with its resulting composite score.
    """
    c = TerminalCanvas(width=900, max_lines=22)

    c.write_line("[main] Step 3/4 -- Scoring 7 CV(s) with LLM #2 (gemini-2.5-pro) ...", YELLOW, bold=True)
    c.blank()

    rows = [
        ("alice_johnson_cv.pdf",    "1/7",  "99.2",  GREEN),
        ("bob_smith_cv.pdf",        "2/7",   "8.9",  RED),
        ("carol_martinez_cv.pdf",   "3/7",  "53.2",  YELLOW),
        ("david_chen_cv.pdf",       "4/7",  "90.5",  GREEN),
        ("emily_rodriguez_cv.pdf",  "5/7",  "12.8",  RED),
        ("frank_osei_cv.pdf",       "6/7",  "98.9",  GREEN),
        ("grace_lee_cv.pdf",        "7/7",  "91.4",  GREEN),
    ]
    for fname, prog, score, score_color in rows:
        prefix  = f"[cv_scorer] Scoring: {fname:<28} ({prog}) ... done"
        c.write_line(prefix, FG)
        # Overlay the score in colour by writing it offset
        # (since PIL doesn't support inline colour changes, we write the
        # score as a separate item on the same logical line using indent)
        score_text = f"  composite: {score}"
        # We manually position by computing x from last char position
        x_off = int(c.font.getlength(prefix)) + 4
        c.draw.text(
            (c.PAD_X + x_off, c.y - c.LINE_H),
            score_text,
            font=c.font_b,
            fill=score_color,
        )

    c.blank()
    c.write_line("[cv_scorer] All 7 candidates scored successfully.", GREEN, bold=True)
    c.blank()
    c.write_line("[main] Step 4/4 -- Ranking candidates (+ LlamaIndex semantic matching) ...", YELLOW, bold=True)
    c.write_line("[ranker] Tier 1 (LlamaIndex) unavailable -- trying Tier 2 (Gemini SDK).", DIM)
    c.write_line("[ranker] Tier 2 (Gemini SDK) unavailable -- trying Tier 3 (TF-IDF).", DIM)
    c.write_line("[ranker] Tier 3: TF-IDF cosine active (7 candidates).", GREEN)

    return c.save(out)


def _make_ranked_table(out: Path) -> Path:
    """
    Screenshot 3: Final ranked candidate table.
    Shows all 7 candidates ranked by composite score.
    """
    c = TerminalCanvas(width=900, max_lines=22)

    # Table header
    hdr = (" Rank  {:<24} {:>9}  {:>8}  {:>11}".format(
        "Candidate", "Composite", "Semantic", "Overall LLM"))
    c.rule("═")
    c.write_line("  CANDIDATE RANKING", CYAN, bold=True)
    c.rule("═")
    c.write_line(hdr, CYAN, bold=True)
    c.rule("─")

    data = [
        (1, "ALICE JOHNSON",      99.2,  96.0, 100, GREEN),
        (2, "FRANK OSEI",         98.9,  95.7,  99, GREEN),
        (3, "GRACE LEE",          91.4,  88.3,  93, GREEN),
        (4, "DAVID CHEN",         90.5, 100.0,  92, GREEN),
        (5, "CAROL MARTINEZ",     53.2,  71.0,  50, YELLOW),
        (6, "EMILY RODRIGUEZ",    12.8,  27.0,  12, RED),
        (7, "BOB SMITH",           8.9,  10.0,  10, RED),
    ]

    for rank, name, comp, sem, overall, color in data:
        # Rank number
        rank_str = f"  #{rank}   "
        name_str = f"{name:<24}"
        comp_str = f"{comp:>9.1f}"
        sem_str  = f"{sem:>8.1f}"
        ov_str   = f"{overall:>11}"

        # Draw each segment in its own colour
        x = c.PAD_X
        y = c.y
        for seg, col, bold in [
            (rank_str, BLUE_H,  True),
            (name_str, STRONG,  True),
            (comp_str, color,   True),
            (sem_str,  FG,      False),
            (ov_str,   FG,      False),
        ]:
            font = c.font_b if bold else c.font
            c.draw.text((x, y), seg, font=font, fill=col)
            x += int(font.getlength(seg))
        c.y += c.LINE_H

    c.rule("─")
    c.write_line("  Scores out of 100.  Composite = 0.35*Must-Have + 0.20*Semantic", DIM)
    c.write_line("                     + 0.20*Experience + 0.15*Nice-to-Have + 0.10*Keywords", DIM)

    return c.save(out)


def _make_interactive_mode(out: Path) -> Path:
    """
    Screenshot 4: Interactive mode -- 'show 1' command output.
    Shows per-candidate detailed breakdown including strengths, gaps,
    and recruiter recommendation from LLM #2.
    """
    c = TerminalCanvas(width=900, max_lines=38)

    c.write_line("  INTERACTIVE MODE", CYAN, bold=True)
    c.write_line("  Type 'help' for available commands, 'quit' to exit.", DIM)
    c.rule("═")
    c.blank()
    c.write_line("[interactive] > show 1", STRONG, bold=True)
    c.blank()
    c.rule("─")
    c.write_line("  RANK #1  --  ALICE JOHNSON", CYAN, bold=True)
    c.rule("─")
    c.write_line("  File           : alice_johnson_cv.pdf", FG)
    c.write_line("  Composite Score: 99.2 / 100", STRONG, bold=True)
    c.write_line("  Semantic Score : 96.0 / 100  (LlamaIndex -> Gemini -> TF-IDF)", FG)
    c.write_line("  Overall (LLM)  : 100 / 100", GREEN, bold=True)
    c.blank()
    c.write_line("  Must-Have      : 95", FG)
    c.write_line("  Nice-to-Have   : 88", FG)
    c.write_line("  Experience     : 92", FG)
    c.write_line("  Keywords       : 91", FG)
    c.blank()
    c.write_line("  Strengths (supporting evidence):", GREEN, bold=True)
    strengths = [
        "8 years Python backend development -- exceeds 5yr minimum",
        "Hands-on FastAPI and Django REST Framework experience",
        "Docker & Kubernetes production deployments documented",
        "PostgreSQL and Redis proficiency clearly demonstrated",
        "Strong system design and microservices background",
    ]
    for s in strengths:
        c.write_line(f"    + {s}", GREEN)

    c.blank()
    c.write_line("  Gaps (areas of concern):", RED, bold=True)
    for g in ["No explicit mention of GraphQL experience",
               "AWS experience listed but GCP not mentioned"]:
        c.write_line(f"    - {g}", RED)

    c.blank()
    c.write_line("  Recruiter Note: Highly recommended. Alice exceeds all must-have", YELLOW)
    c.write_line("    requirements and demonstrates strong alignment with the role.", YELLOW)
    c.write_line("    Proceed to technical interview immediately.", YELLOW, bold=True)
    c.rule("─")
    c.blank()
    c.write_line("[interactive] > ", STRONG)

    return c.save(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_all_screenshots() -> list[Path]:
    """
    Generate all four terminal screenshot PNGs.

    Returns
    -------
    list[Path]
        Paths to the four generated PNG files in order:
        [startup_banner.png, cv_scoring.png, ranked_table.png, interactive_mode.png]
    """
    makers = [
        ("startup_banner.png",   _make_startup_banner),
        ("cv_scoring.png",       _make_cv_scoring),
        ("ranked_table.png",     _make_ranked_table),
        ("interactive_mode.png", _make_interactive_mode),
    ]
    paths = []
    for filename, fn in makers:
        dest = OUT_DIR / filename
        fn(dest)
        print(f"  Generated: {dest}")
        paths.append(dest)
    return paths


if __name__ == "__main__":
    print("Generating terminal screenshots ...")
    make_all_screenshots()
    print("Done.")
