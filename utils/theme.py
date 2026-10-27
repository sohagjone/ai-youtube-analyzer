"""Visual theme: palette, CSS and small HTML helpers for a colourful, professional dashboard."""
from __future__ import annotations

import html

PALETTE = {
    "violet": "#6C5CE7", "teal": "#00B894", "pink": "#FD79A8", "amber": "#FDCB6E", "blue": "#0984E3",
    "coral": "#E17055", "navy": "#2D3436", "slate": "#636E72", "mist": "#F5F6FF",
}
COLORWAY = ["#6C5CE7", "#00B894", "#FD79A8", "#0984E3", "#FDCB6E", "#E17055", "#A29BFE", "#55EFC4"]
GRADIENTS = [
    "linear-gradient(135deg,#6C5CE7,#A29BFE)", "linear-gradient(135deg,#00B894,#55EFC4)",
    "linear-gradient(135deg,#0984E3,#74B9FF)", "linear-gradient(135deg,#FD79A8,#E17055)",
    "linear-gradient(135deg,#F79F1F,#FDCB6E)", "linear-gradient(135deg,#2D3436,#636E72)",
]

CSS = """
<style>
.block-container {padding-top: 3.2rem; max-width: 1400px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#2B2358 0%,#4B3FB5 55%,#6C5CE7 100%);}
[data-testid="stSidebar"] * {color: #fff !important;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {color: #2D3436 !important;}
[data-testid="stSidebar"] [data-baseweb="input"] {background: #fff;}
[data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg,#FD79A8,#E17055); border: 0; color: #fff !important; font-weight: 700; border-radius: 10px;}
.hero {background: linear-gradient(120deg,#6C5CE7 0%,#0984E3 55%,#00B894 100%); color:#fff; padding: 26px 30px; border-radius: 18px;
  margin-bottom: 18px; box-shadow: 0 10px 30px rgba(108,92,231,.25);}
.hero h1 {color:#fff; margin:0; font-size: 2.0rem; font-weight: 800;}
.hero p {color: rgba(255,255,255,.92); margin: 6px 0 0 0; font-size: 1.02rem;}
.kpi {border-radius: 16px; padding: 16px 18px; color:#fff; box-shadow: 0 6px 18px rgba(45,52,54,.15); min-height: 104px;}
.kpi .l {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .92;}
.kpi .v {font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin-top: 4px;}
.kpi .s {font-size: .78rem; opacity: .9; margin-top: 2px;}
.sec {border-left: 6px solid #6C5CE7; padding: 4px 0 4px 12px; margin: 22px 0 10px 0; font-size: 1.25rem; font-weight: 800; color:#2D3436;}
.card {background:#fff; border:1px solid #E6E8FF; border-left: 6px solid var(--c,#6C5CE7); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
  box-shadow: 0 3px 10px rgba(108,92,231,.08);}
.card h4 {margin: 0 0 4px 0; color:#2D3436;}
.card .meta {color:#636E72; font-size:.86rem;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; color:#fff; margin-right: 6px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {background:#F0F1FF; border-radius: 10px 10px 0 0; padding: 8px 14px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg,#6C5CE7,#0984E3); color:#fff !important;}
.stTabs [aria-selected="true"] p {color:#fff !important;}
div[data-testid="stDownloadButton"] button {background: linear-gradient(135deg,#00B894,#0984E3); color:#fff; border:0; border-radius: 10px; font-weight: 700;}
div[data-testid="stExpander"] {border: 1px solid #E6E8FF; border-radius: 12px; background:#FBFBFF;}
</style>
"""


def hero(title: str, subtitle: str) -> str:
    return f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'


def kpi_card(label: str, value: str, sub: str = "", gradient: int = 0) -> str:
    return (f'<div class="kpi" style="background:{GRADIENTS[gradient % len(GRADIENTS)]}"><div class="l">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div><div class="s">{html.escape(sub)}</div></div>')


def section(title: str) -> str:
    return f'<div class="sec">{html.escape(title)}</div>'


def card(title: str, body_html: str, color: str = "#6C5CE7", badge: str = "") -> str:
    b = f'<span class="badge" style="background:{color}">{html.escape(badge)}</span>' if badge else ""
    return f'<div class="card" style="--c:{color}"><h4>{b}{html.escape(title)}</h4><div class="meta">{body_html}</div></div>'
