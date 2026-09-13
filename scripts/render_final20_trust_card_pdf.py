"""Render the final20 Trust Card markdown and figures into a polished PDF."""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "deliverables_final"
MD = DEST / "TRUST_CARD.md"
EVIDENCE = DEST / "final20_evidence.json"
FIG = DEST / "figures_final20"
OUT = ROOT / "output" / "pdf" / "TRUST_CARD_FINAL20.pdf"
PACKAGE_COPY = DEST / "TRUST_CARD_FINAL20.pdf"


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.setLineWidth(0.5)
    canvas.line(0.7 * inch, 0.52 * inch, 7.8 * inch, 0.52 * inch)
    canvas.setFillColor(colors.HexColor("#4b5563"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.7 * inch, 0.34 * inch, "ML and AI Nexus 2026 - Final20 Trust Card")
    canvas.drawRightString(7.8 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=colors.black, alignment=TA_LEFT, spaceAfter=12))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, textColor=colors.HexColor("#374151"), spaceAfter=16))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.black, spaceBefore=14, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.black, spaceBefore=10, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.2, textColor=colors.HexColor("#111827"), spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="BulletCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.8, leftIndent=14, firstLineIndent=-9, textColor=colors.HexColor("#111827"), spaceAfter=3))
    styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8.2, leading=11, textColor=colors.HexColor("#4b5563"), alignment=TA_CENTER, spaceBefore=3, spaceAfter=9))
    return styles


def make_metric_table(evidence):
    m = evidence["metrics"]
    rows = [
        ["Metric", "Final20 value", "Context"],
        ["Nested OOF log loss", f"{m['log_loss']:.7f}", "Lower is better"],
        ["Public Kaggle score", f"{evidence['public_kaggle_score']:.5f}", "User-reported leaderboard result"],
        ["Brier score", f"{m['brier']:.7f}", "Probability accuracy"],
        ["ROC-AUC", f"{m['roc_auc']:.5f}", "Ranking discrimination"],
        ["Outer validation", f"{evidence['outer_folds']} folds", f"{evidence['training_fraction']:.0%} training per model"],
    ]
    table = Table(rows, colWidths=[1.75 * inch, 1.35 * inch, 3.95 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def add_figure(story, filename, caption, width=6.6 * inch):
    path = FIG / filename
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    story.append(KeepTogether([image, Paragraph(caption, make_styles()["CaptionCustom"])]))


def build_story():
    styles = make_styles()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    md = MD.read_text(encoding="utf-8")
    story = []
    story.append(Paragraph("Trust Card", styles["CoverTitle"]))
    story.append(Paragraph("ML and AI Nexus 2026 Final20 Model", styles["CoverSub"]))
    story.append(Paragraph("Current champion for the competition workflow", styles["BodyCustom"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(make_metric_table(evidence))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Scope", styles["H2Custom"]))
    story.append(Paragraph("This card documents the frozen final20 calibrated ensemble. The untested 20-fold five-seed candidate is not selected, and v6-derived outputs are excluded from this package.", styles["BodyCustom"]))
    story.append(PageBreak())

    # Render the markdown body with a compact, deterministic markdown subset.
    in_figure_index = False
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["H2Custom"]))
            in_figure_index = True
            continue
        if line.startswith("## "):
            title = line[3:]
            story.append(Paragraph(inline_markup(title), styles["H1Custom"]))
            in_figure_index = False
            continue
        if line.startswith("- "):
            story.append(Paragraph("&bull;&nbsp;&nbsp;" + inline_markup(line[2:]), styles["BulletCustom"]))
            continue
        if line.startswith("**Decision:**"):
            story.append(Paragraph(inline_markup(line), styles["BodyCustom"]))
            continue
        story.append(Paragraph(inline_markup(line), styles["BodyCustom"]))

    # Put the figures after the narrative so the evidence is easy to find and review.
    story.append(PageBreak())
    story.append(Paragraph("Final20 figures", styles["H1Custom"]))
    story.append(Paragraph("All figures below are generated from the final20 OOF predictions and supplied training labels by `scripts/final20_trust_card_figures.py`.", styles["BodyCustom"]))
    add_figure(story, "final20_calibration.png", "Figure 1. Calibration curves and probability distribution for raw and calibrated equal-weight OOF predictions.")
    add_figure(story, "final20_fold_stability.png", "Figure 2. Held-out binary log loss for each of the 20 outer folds.")
    add_figure(story, "final20_threshold_tradeoff.png", "Figure 3. Sensitivity and specificity over reporting thresholds; the dashed line marks 0.15.")
    add_figure(story, "final20_subgroup_logloss.png", "Figure 4. Final20 OOF log loss by observed subgroup, with subgroup size and prevalence labels.")
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="ML and AI Nexus 2026 Final20 Trust Card",
        author="OpenAI Codex",
    )
    doc.build(build_story(), onFirstPage=page_footer, onLaterPages=page_footer)
    shutil.copy2(OUT, PACKAGE_COPY)
    print(OUT)


if __name__ == "__main__":
    main()
