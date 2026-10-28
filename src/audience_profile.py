"""Channel 'audience behaviour' profile built ONLY from public data.

What the audience really watches, when they are online, how long they stay and who they are is private to the channel owner
(YouTube Studio). Here 'behaviour' means what the public numbers show: which formats, lengths, title styles, topics, publish
slots and hashtags went with more views, likes and comments on THIS channel. These are correlations, not causes.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

SHORT_FMT = "Short-form (<=3 min)"
LONG_FMT = "Long-form (>3 min)"
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_\u00C0-\uFFFF]{2,30})")
HOUR_BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
SHRINK_K = 5  # pseudo-observations pulling small groups toward 1.0 (no effect)


def hour_block_label(h: int) -> str:
    a, b = next((a, b) for a, b in HOUR_BLOCKS if a <= h < b)
    return f"{a:02d}:00-{b:02d}:00"


def mature_with_format_relative(df: pd.DataFrame, min_age: float = 7) -> pd.DataFrame:
    """Copy of df with `rel_fmt` = views / median views of the same format, using videos old enough to have settled."""
    d = df.copy()
    if d.empty:
        d["rel_fmt"] = pd.Series(dtype=float)
        return d
    older = d[d["age_days"] >= min_age]
    d = older if len(older) >= 10 else d
    med = d.groupby("format")["views"].transform("median").replace(0, np.nan)
    d["rel_fmt"] = d["views"] / med
    return d


def shrunk_effects(d: pd.DataFrame, col: str, k: int = SHRINK_K, value: str = "rel_fmt") -> pd.DataFrame:
    """Median relative performance per group, pulled toward 1.0 when the group is small. `effect` is what to trust."""
    if d.empty or col not in d:
        return pd.DataFrame(columns=["group", "videos", "median_relative", "effect"])
    rows = []
    for grp, sub in d.groupby(col, observed=True):
        n, med = len(sub), float(sub[value].median())
        rows.append({"group": grp, "videos": n, "median_relative": med, "effect": 1.0 + (med - 1.0) * n / (n + k)})
    return pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)


def extract_hashtags(*texts: str) -> list[str]:
    seen: list[str] = []
    for t in texts:
        seen += [h.lower() for h in HASHTAG_RE.findall(t or "")]
    return seen


def reliability(n: int) -> str:
    return "high" if n >= 40 else "medium" if n >= 20 else "low" if n >= 8 else "very low"


def build_profile(df: pd.DataFrame, tz_offset_hours: float = 0.0) -> dict:
    """Everything the studio features need to know about this channel. Safe on tiny or empty samples."""
    if df is None or df.empty or "format" not in df:
        return {"available": False}
    d = mature_with_format_relative(df)
    pub = pd.to_datetime(d["published_at"], utc=True) + pd.to_timedelta(tz_offset_hours, unit="h")
    d = d.assign(weekday=pub.dt.day_name(), hour_block=pub.dt.hour.map(hour_block_label), weekend=np.where(pub.dt.dayofweek >= 5, "weekend", "weekday"))
    n_short = int((df["format"] == SHORT_FMT).sum())
    tags = Counter(h for t, ds in zip(df["title"], df["description"]) for h in set(extract_hashtags(t, ds)))
    tag_list = Counter(str(t).lower() for ts in df["tags"] for t in (ts or []))
    own_hashtags = [h for h, c in tags.most_common(15) if c >= 2]
    eff = {c: shrunk_effects(d, c) for c in ("duration_bucket", "title_length_bucket", "title_pattern", "topic", "weekday", "hour_block", "weekend", "format")
           if c in d}
    best = {c: (e.iloc[0]["group"] if len(e) and e.iloc[0]["effect"] > 1.0 else None) for c, e in eff.items()}
    shorts, longs = df[df["format"] == SHORT_FMT], df[df["format"] == LONG_FMT]
    eng = lambda x, c: float(x[c].median()) if len(x) and x[c].notna().any() else np.nan  # noqa: E731
    return {
        "available": True, "n_videos": len(df), "n_short": n_short, "n_long": len(df) - n_short, "reliability": reliability(len(d)),
        "effects": eff, "best": best, "tz_offset_hours": tz_offset_hours,
        "hashtags_used": own_hashtags, "tags_used": [t for t, c in tag_list.most_common(15)],
        "engagement": {"short_like_rate": eng(shorts, "like_rate"), "long_like_rate": eng(longs, "like_rate"),
                       "short_comment_rate": eng(shorts, "comment_rate"), "long_comment_rate": eng(longs, "comment_rate")},
        "median_views": {"short": float(shorts["views"].median()) if len(shorts) else np.nan, "long": float(longs["views"].median()) if len(longs) else np.nan},
        "median_title_chars": float(df["title_chars"].median()) if "title_chars" in df else np.nan,
        "median_duration_s": {"short": float(shorts["duration_seconds"].median()) if len(shorts) else np.nan,
                              "long": float(longs["duration_seconds"].median()) if len(longs) else np.nan},
        "note": "Built from public numbers only (views, likes, comments, titles, publish times). Real audience watch time, retention, "
                "demographics and online hours are private to the channel owner (YouTube Studio > Analytics > Audience).",
    }
