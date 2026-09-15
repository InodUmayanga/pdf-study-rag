"""Generate the evaluation fixture PDF from ``fixture_text.PAGES``.

The PDF is rendered with PyMuPDF on demand and is never committed, so the
repository carries no study material — only the original text in
``fixture_text.py``. The ingestion pipeline OCRs every page regardless of
whether it has a text layer, so this born-digital PDF still exercises the
real render -> OCR -> embed path.

Usage (from the repository root):

    python -m evals.make_fixture            # writes evals/.tmp/fixture/
    python -m evals.make_fixture --out DIR  # somewhere else
"""

import argparse
import os
import sys

import pymupdf

# Allow ``python evals/make_fixture.py`` as well as ``python -m ...``.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIXTURE_NAME = "sample_notes.pdf"
DEFAULT_OUT_DIR = os.path.join(ROOT, "evals", ".tmp", "fixture")

# A4 in PDF points, with generous margins so OCR sees clean text.
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 56
TITLE_FONTSIZE = 16
BODY_FONTSIZE = 13


def fixture_pages():
    """The ``(title, body)`` pairs the fixture is built from."""
    from evals.fixture_text import PAGES

    return list(PAGES)


def write_fixture(out_dir=DEFAULT_OUT_DIR, pages=None):
    """Render the fixture PDF into ``out_dir``; returns the PDF path.

    Always regenerates, so the PDF can never drift from ``fixture_text``.
    """
    pages = pages if pages is not None else fixture_pages()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FIXTURE_NAME)

    doc = pymupdf.open()
    try:
        for number, (title, body) in enumerate(pages, 1):
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            title_rect = pymupdf.Rect(
                MARGIN, MARGIN, PAGE_WIDTH - MARGIN, MARGIN + 40
            )
            body_rect = pymupdf.Rect(
                MARGIN,
                MARGIN + 50,
                PAGE_WIDTH - MARGIN,
                PAGE_HEIGHT - MARGIN,
            )
            leftover = page.insert_textbox(
                title_rect, title, fontsize=TITLE_FONTSIZE, fontname="helv"
            )
            if leftover < 0:
                raise ValueError(f"title of page {number} does not fit")
            leftover = page.insert_textbox(
                body_rect, body, fontsize=BODY_FONTSIZE, fontname="helv"
            )
            if leftover < 0:
                raise ValueError(f"body of page {number} does not fit")
        doc.save(path)
    finally:
        doc.close()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    path = write_fixture(args.out)
    print(f"Wrote {path} ({len(fixture_pages())} pages)")


if __name__ == "__main__":
    main()
