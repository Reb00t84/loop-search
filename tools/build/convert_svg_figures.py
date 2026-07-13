#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts fig1_penrose_loop(_ru).svg and fig2_parameter_plane(_ru).svg to PDF
for the preprint build (build_preprint.py expects these as .pdf, but only
.svg sources exist on disk - the PDFs were apparently made by hand once and
never kept).

No native cairo library is installed on this machine (cairosvg needs it and
fails to load), so this uses svglib+reportlab instead (pure Python, no
native deps). Caveat found the hard way: the SVGs set font-family=
"Helvetica,Arial,sans-serif", and ReportLab's built-in Helvetica is a core
PDF font with no Cyrillic glyphs - the RU figures rendered as solid black
boxes where the text should be. Fix: substitute that font-family string for
a custom name registered to DejaVu Sans (bundled with matplotlib, already a
project dependency, no extra install needed) before parsing. Applied
uniformly to EN and RU so all four figures use the same font.

Usage: python3 tools/build/convert_svg_figures.py
Writes into "preprint files/" (gitignored, not part of the repo).
"""
import os
from svglib import svglib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import matplotlib

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIG_DIR = os.path.join(ROOT, "preprint files")
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
DEJAVU = os.path.join(FONT_DIR, "DejaVuSans.ttf")
DEJAVU_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
OLD_FONT = "Helvetica,Arial,sans-serif"
NEW_FONT = "DejaVuCyr"

FILES = ["fig1_penrose_loop", "fig1_penrose_loop_ru",
         "fig2_parameter_plane", "fig2_parameter_plane_ru"]


def main():
    svglib.register_font(NEW_FONT, DEJAVU)
    # SVGs use font-weight="500" on the axis-title text elements. svglib's
    # internal font key is built as f"{family}-{weight.capitalize()}", so
    # weight="500" needs its OWN registration ("DejaVuCyr-500") - weight=
    # "bold" registers a different key ("DejaVuCyr-Bold") and does not
    # cover it, which is why that attempt still rendered boxes.
    svglib.register_font(NEW_FONT, DEJAVU_BOLD, weight="500")
    svglib.register_font(NEW_FONT, DEJAVU_BOLD, weight="bold")
    for name in FILES:
        svg_path = os.path.join(FIG_DIR, name + ".svg")
        pdf_path = os.path.join(FIG_DIR, name + ".pdf")
        src = open(svg_path, encoding="utf-8").read()
        n = src.count(OLD_FONT)
        assert n >= 1, f"{name}.svg: expected font-family {OLD_FONT!r} not found"
        tmp_path = svg_path + ".tmp"
        open(tmp_path, "w", encoding="utf-8").write(src.replace(OLD_FONT, NEW_FONT))
        try:
            drawing = svg2rlg(tmp_path)
            renderPDF.drawToFile(drawing, pdf_path)
        finally:
            os.remove(tmp_path)
        print(name, "->", pdf_path)


if __name__ == "__main__":
    main()
