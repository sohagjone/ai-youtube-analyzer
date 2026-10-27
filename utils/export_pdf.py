"""PDF report (reportlab). Built from the same analysis results as every other export."""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.charts import build_charts
from utils.data_processing import build_export_tables, format_number, prepare_df_for_export

RED = colors.HexColor("#d62828")
MAX_ROWS = 25
MAX_CHARS = 140


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    s = str(v)
    return s if len(s) <= MAX_CHARS else s[: MAX_CHARS - 1] + "..."


def _table(df: pd.DataFrame, styles, width: float, cols: list[str] | None = None, max_rows: int = MAX_ROWS):
    df = prepare_df_for_export(df)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    df = df.head(max_rows)
    small = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=7, leading=8.5)
    head = ParagraphStyle("head", parent=small, textColor=colors.white, fontName="Helvetica-Bold")
    data = [[Paragraph(escape(str(c)), head) for c in df.columns]]
    for row in df.itertuples(index=False):
        data.append([Paragraph(escape(_fmt(v)), small) for v in row])
    col_w = width / max(len(df.columns), 1)
    t = Table(data, colWidths=[col_w] * len(df.columns), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RED), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd3da")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_pdf(results: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="AI YouTube Channel Analyzer report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=RED)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#1f2430"))
    body = styles["BodyText"]
    width = A4[0] - 3.2 * cm
    meta = results.get("meta", {})
    channel = results.get("channel", {})
    tables = build_export_tables(results)
    charts = build_charts(results)
    story: list = [Paragraph("AI YouTube Channel Analyzer", h1),
                   Paragraph(escape(f"Channel: {channel.get('title', '')}"), styles["Heading3"]),
                   Paragraph(escape(f"Analysis date (UTC): {str(meta.get('analysis_date', ''))[:19]} | Data mode: {meta.get('mode', '')}"), body)]
    if str(meta.get("mode", "")).startswith("SAMPLE"):
        story.append(Paragraph("<b>SAMPLE DATA: every figure in this report is illustrative and not real YouTube data.</b>", body))
    story.append(Spacer(1, 6))

    def section(title: str, key: str | None, chart: str | None = None, cols: list[str] | None = None, note: str | None = None):
        story.append(Paragraph(title, h2))
        if note:
            story.append(Paragraph(escape(note), body))
        if chart and chart in charts:
            story.append(Image(io.BytesIO(charts[chart]), width=width, height=width * 0.43))
        if key and key in tables and len(tables[key]):
            story.append(Spacer(1, 4))
            story.append(_table(tables[key], styles, width, cols))
        elif not chart:
            story.append(Paragraph("No data available for this section.", body))
        story.append(Spacer(1, 8))

    section("1. Channel overview", "Channel Overview", note="Lifetime figures describe the whole channel; sample figures describe only the selected videos/date range.")
    section("2. Video performance", "Video Performance", "Views timeline",
            cols=["Title", "Published (UTC)", "Views (public)", "Views per day", "Relative to median", "Format", "Performance flag"],
            note="Public view counts only. Older videos have had longer to accumulate views; low views are not automatically a failure.")
    section("3. Title analysis", "Title Patterns", cols=["Feature", "Group", "Videos", "Median Views", "Median Relative", "Note"],
            note="Patterns are observational and not proof that a style causes better results. Thumbnail click-through rate is not available from public data.")
    section("4. Topic breakdown", "Topic Analysis", "Topics", cols=["Topic", "Videos", "Share", "Median Views", "Median Relative", "Observation"],
            note=f"Topic method: {meta.get('topic_method', '')}. Based on metadata (titles, tags, descriptions), not full video content.")
    section("5. Competitor comparison", "Competitors", "Competitors",
            cols=["Role", "Channel", "Subscribers", "Median Views", "Uploads Per 30 Days", "Short Form Share", "Main Topics"])
    if results.get("competitor_differences"):
        for line in results["competitor_differences"]:
            story.append(Paragraph("&bull; " + escape(line), body))
    section("6. Content gap recommendations", "Content Gaps", "Content gaps",
            cols=["Rank", "Gap Type", "Topic", "Suggested Title", "Format", "Priority Score", "Confidence"],
            note="Priority = 30% relevance + 30% demand evidence + 20% differentiation + 20% feasibility. A content gap is not proof of search demand. Suggested titles are recommendations, not guaranteed improvements.")
    section("7. Content calendar", "Content Calendar", cols=["Date", "Format", "Topic", "Working Title", "Basis"])
    story.append(PageBreak())
    story.append(Paragraph("8. Sources, methodology and limitations", h2))
    for s in results.get("sources", []):
        story.append(Paragraph(escape(f"{s['type']}: {s['item']} - {s['detail']}"), body))
    story.append(Spacer(1, 6))
    for i, lim in enumerate(results.get("limitations", []), 1):
        story.append(Paragraph(escape(f"{i}. {lim}"), body))
    doc.build(story)
    return buf.getvalue()
