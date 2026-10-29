"""Excel export for one Video Studio run (editing advice, titles, planner, risk screen)."""
from __future__ import annotations

import io

import pandas as pd
from openpyxl import Workbook

from utils.export_excel import _write_sheet


def build_studio_workbook(st: dict) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    summary = [{"Item": "Planned format", "Value": st.get("format")}, {"Item": "Working title", "Value": st["inputs"].get("title")},
               {"Item": "Copyright/reuse risk (screen only)", "Value": f"{st['copyright']['risk_score']}/100 ({st['copyright']['level']})"},
               {"Item": "Disclaimer", "Value": st["copyright"]["disclaimer"]}]
    _write_sheet(wb, "Summary", pd.DataFrame(summary))
    if len(st.get("editing_advice", [])):
        _write_sheet(wb, "Editing Advice", st["editing_advice"].rename(columns=lambda c: c.replace("_", " ").title()))
    if len(st.get("titles", [])):
        _write_sheet(wb, "Titles", st["titles"].rename(columns=lambda c: c.replace("_", " ").title()))
    cap = [{"Item": "Caption / description draft", "Value": st.get("caption", "")}, {"Item": "Hashtags", "Value": " ".join(st["hashtags"]["hashtags"])},
           {"Item": "Shown above the title", "Value": " ".join(st["hashtags"]["above_title"])}, {"Item": "Hidden tags", "Value": st["hashtags"]["hidden_tags"]}]
    _write_sheet(wb, "Caption and Hashtags", pd.DataFrame(cap))
    u = st.get("understanding", {})
    if u:
        _write_sheet(wb, "AI Understanding", pd.DataFrame([{"Item": k.replace("_", " ").title(), "Value": ("; ".join(v) if isinstance(v, list) else v)} for k, v in u.items()]))
    fc = st.get("forecast", {})
    if fc.get("available"):
        t = fc["table"].rename(columns={"metric": "Metric", "low": "Low", "mid": "Typical", "high": "High", "basis": "Basis"}).round(1)
        notes = pd.DataFrame([{"Metric": "Note", "Basis": n} for n in fc["notes"]])
        _write_sheet(wb, "Predicted Performance", pd.concat([t, notes], ignore_index=True))
    if len(st.get("comparison", [])):
        _write_sheet(wb, "Competitor Comparison", st["comparison"].rename(columns=lambda c: c.replace("_", " ").title()))
    if len(st.get("closest", [])):
        _write_sheet(wb, "Closest Competitor Videos", st["closest"].rename(columns=lambda c: c.replace("_", " ").title()))
    pl = st.get("planner", {})
    if len(pl.get("plan", [])):
        _write_sheet(wb, "Upload Plan", pl["plan"].rename(columns=lambda c: c.replace("_", " ").title()))
    if len(pl.get("best", [])):
        _write_sheet(wb, "Best Slots", pl["best"].drop(columns=["priority"], errors="ignore").rename(columns=lambda c: c.replace("_", " ").title()))
    f = st["copyright"]["findings"]
    _write_sheet(wb, "Copyright Risk", f.rename(columns=lambda c: c.replace("_", " ").title()) if len(f) else pd.DataFrame([{"Result": st["copyright"]["summary"]}]))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
