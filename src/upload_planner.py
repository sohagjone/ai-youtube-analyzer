"""Upload planner: best publish slots from the channel's own history, then a dated schedule.

Slot scores come from public views of this channel's own videos grouped by weekday and by 4-hour block (shrunk toward 'no effect' when
a group is small). They show when past videos that did well were PUBLISHED, which is not the same as when the audience is online:
check YouTube Studio > Analytics > Audience ('When your viewers are on YouTube') and treat that as the final word.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.audience_profile import HOUR_BLOCKS, LONG_FMT, SHORT_FMT, build_profile, hour_block_label
from src.content_strategy import generate_calendar

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CHECKLIST = [
    "Before publishing: title, thumbnail (long-form) and first 2 seconds agree on the same promise.",
    "Run YouTube Studio's copyright and ad-suitability checks during upload.",
    "First hour: pin a comment with a question and answer early replies.",
    "First 24 hours: share to one place your audience actually uses (community post, newsletter), without spamming.",
    "After 48 hours: note views, likes per view and, in Studio, average view duration; log them next to this plan.",
]


def slot_table(df: pd.DataFrame, tz_offset_hours: float = 0.0) -> dict:
    prof = build_profile(df, tz_offset_hours)
    if not prof.get("available"):
        return {"available": False}
    wd = prof["effects"].get("weekday", pd.DataFrame()).rename(columns={"group": "weekday"})
    hb = prof["effects"].get("hour_block", pd.DataFrame()).rename(columns={"group": "hour_block"})
    if wd.empty or hb.empty:
        return {"available": False}
    wmap, hmap = dict(zip(wd["weekday"], wd["effect"])), dict(zip(hb["hour_block"], hb["effect"]))
    wn, hn = dict(zip(wd["weekday"], wd["videos"])), dict(zip(hb["hour_block"], hb["videos"]))
    rows = []
    for d in DAYS:
        for a, b in HOUR_BLOCKS:
            lbl = hour_block_label(a)
            rows.append({"weekday": d, "hour_block": lbl, "score": wmap.get(d, 1.0) * hmap.get(lbl, 1.0), "weekday_videos": wn.get(d, 0), "block_videos": hn.get(lbl, 0)})
    slots = pd.DataFrame(rows)
    slots["evidence"] = np.where((slots["weekday_videos"] >= 4) & (slots["block_videos"] >= 6), "ok", "thin")
    return {"available": True, "slots": slots.sort_values("score", ascending=False).reset_index(drop=True), "weekday": wd, "hour_block": hb,
            "reliability": prof["reliability"], "tz_offset_hours": tz_offset_hours, "n": prof["n_videos"]}


def format_ranges(df: pd.DataFrame) -> dict:
    """Typical settled views per format (25th/50th/75th percentile of videos at least 14 days old)."""
    out = {}
    for key, name in (("short", SHORT_FMT), ("long", LONG_FMT)):
        s = df[(df["format"] == name) & (df["age_days"] >= 14)]["views"]
        out[key] = (float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))) if len(s) >= 3 else None
    return out


def build_plan(df: pd.DataFrame, gaps, topics, start: date | None = None, days: int = 14, posts_per_week: int = 3, formats: list[str] | None = None,
               niche: str = "", audience: str = "", tz_offset_hours: float = 0.0, minimum_gap_days: int = 1) -> pd.DataFrame:
    """Schedule ideas onto the best weekdays/hours. Uses the existing evidence-labelled idea pool and assigns each a slot + expected range."""
    start = start or date.today()
    st = slot_table(df, tz_offset_hours)
    cal = generate_calendar(gaps, topics, days=days, posts_per_week=posts_per_week, formats=formats, niche=niche, audience=audience, start=start)
    if cal.empty:
        return cal
    ranges = format_ranges(df)
    if st["available"]:
        best_per_day = st["slots"].groupby("weekday").head(1).set_index("weekday")
        order = [d for d in st["slots"]["weekday"].drop_duplicates()]
    else:
        best_per_day, order = None, DAYS
    window = [start + timedelta(days=i) for i in range(days)]
    free = {d: True for d in window}
    used: list[date] = []
    rows = []
    for item in cal.itertuples():
        # choose the highest-scoring weekday with an open date at least `minimum_gap_days` from earlier posts
        pick = None
        for wd in order:
            for d in window:
                if free[d] and d.strftime("%A") == wd and all(abs((d - u).days) >= minimum_gap_days for u in used):
                    pick = d
                    break
            if pick:
                break
        if pick is None:
            pick = next((d for d in window if free[d]), start)
        free[pick] = False
        used.append(pick)
        short = item.format == "YouTube Short"
        rng = ranges["short" if short else "long"]
        slot = best_per_day.loc[pick.strftime("%A")] if best_per_day is not None else None
        score = float(slot["score"]) if slot is not None else 1.0
        rows.append({"date": pick.isoformat(), "weekday": pick.strftime("%A"), "best_time_block": slot["hour_block"] if slot is not None else "Check YouTube Studio audience hours",
                     "slot_score": round(score, 2), "slot_evidence": slot["evidence"] if slot is not None else "none", "format": item.format, "topic": item.topic,
                     "working_title": item.working_title, "opening_hook": item.opening_hook, "basis": item.basis,
                     "typical_views_low": round(rng[0] * score) if rng else None, "typical_views_mid": round(rng[1] * score) if rng else None,
                     "typical_views_high": round(rng[2] * score) if rng else None})
    plan = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    plan["time_zone"] = f"UTC{tz_offset_hours:+g}"
    return plan
