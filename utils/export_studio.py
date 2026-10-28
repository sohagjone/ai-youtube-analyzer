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
    sh = st.get("shorts", {})
    if sh.get("available"):
        rows = [{"Metric": "Baseline (median of your mature Shorts)", "Value": round(sh["baseline"])}, {"Metric": "Estimate low", "Value": round(sh["estimate_low"])},
                {"Metric": "Estimate mid", "Value": round(sh["estimate_mid"])}, {"Metric": "Estimate high", "Value": round(sh["estimate_high"])},
                {"Metric": "Confidence", "Value": sh["confidence"]}, {"Metric": "Method", "Value": sh["method"]}] + [{"Metric": "Note", "Value": n} for n in sh["notes"]]
        _write_sheet(wb, "Shorts Estimate", pd.DataFrame(rows))
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
