"""Trend Radar: YouTube's own most-popular chart plus 'high-potential' recent videos from creators you choose.

Scope and compliance: this uses ONLY the official YouTube Data API v3 (public data). It does not scrape YouTube or any
other platform. TikTok, Instagram, X and others do not offer public APIs for this, and scraping them breaks their terms,
so they are intentionally not supported.
'''High potential''' = a recent video whose pace (views per day) is well above that creator's own usual pace.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import video_analysis as va
from src.title_analysis import classify_pattern
from src.youtube_api import YouTubeAPIError
from utils.data_processing import format_duration, top_terms, videos_to_df
from utils.validation import parse_channel_input

PLATFORM_NOTE = ("Only YouTube is supported, through the official YouTube Data API. Other platforms (TikTok, Instagram, X, Facebook...) "
                 "have no public API for this and scraping them violates their terms, so the app does not scrape them.")
MAX_CREATORS = 10
BREAKOUT, ABOVE = 2.5, 1.5  # pace ratio thresholds (video views/day vs the creator's own median views/day)

TREND_CATEGORIES = {"All categories": None, "Gaming": "20", "Music": "10", "Sports": "17", "Entertainment": "24", "Education": "27",
                    "Science & Technology": "28", "Howto & Style": "26", "People & Blogs": "22", "Comedy": "23", "News & Politics": "25"}


def _prep(videos: list[dict], now: datetime) -> pd.DataFrame:
    df = va.enrich_videos(videos_to_df(videos), now)
    if df.empty:
        return df
    df["views_per_hour"] = df["views"] / (df["age_days"].clip(lower=1 / 24) * 24)
    df["pattern"] = df["title"].apply(classify_pattern)
    df["duration"] = df["duration_seconds"].apply(format_duration)
    return df


def trending_chart(client, region: str = "US", category_id: str | None = None, n: int = 25, now: datetime | None = None) -> pd.DataFrame:
    """YouTube's most-popular chart, enriched with pace metrics. Raises YouTubeAPIError (e.g. no chart for that category)."""
    now = now or datetime.now(timezone.utc)
    df = _prep(client.get_most_popular(region, category_id, n), now)
    if df.empty:
        return df
    return df.sort_values("views_per_hour", ascending=False).reset_index(drop=True)


def creator_radar(client, channel_inputs: list[str], per_channel: int = 30, window_days: int = 30, now: datetime | None = None,
                  progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Recent videos of the given creators ranked by pace vs each creator's own usual pace. Returns (table, warnings)."""
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    rows: list[dict] = []
    inputs = [c for c in channel_inputs if c and c.strip()][:MAX_CREATORS]
    if len([c for c in channel_inputs if c and c.strip()]) > MAX_CREATORS:
        warnings.append(f"Only the first {MAX_CREATORS} creators are analysed.")
    for i, text in enumerate(inputs):
        if progress:
            progress(i / max(len(inputs), 1), f"Reading creator {i + 1} of {len(inputs)}...")
        try:
            ref = parse_channel_input(text)
            bundle = client.fetch_channel_bundle(ref, per_channel)
        except (ValueError, YouTubeAPIError) as exc:
            warnings.append(f"{text.strip()}: {exc}")
            continue
        df = _prep(bundle["videos"], now)
        if len(df) < 4:
            warnings.append(f"{bundle['channel'].get('title') or text}: too few public videos to judge pace.")
            continue
        mature = df[df["age_days"] >= 14]
        base = float((mature if len(mature) >= 3 else df)["views_per_day"].median())
        if base <= 0:
            continue
        recent = df[df["age_days"] <= window_days]
        for r in recent.itertuples():
            ratio = float(r.views_per_day / base)
            status = "Breakout" if ratio >= BREAKOUT else "Above usual pace" if ratio >= ABOVE else "Usual pace" if ratio >= 0.7 else "Below usual pace"
            rows.append({"creator": bundle["channel"].get("title") or text, "title": r.title, "url": r.url, "published_at": r.published_at,
                         "age_days": round(float(r.age_days), 1), "views": int(r.views), "views_per_day": float(r.views_per_day),
                         "views_per_hour": float(r.views_per_hour), "creator_usual_views_per_day": base, "pace_ratio": round(ratio, 2),
                         "status": status, "format": r.format, "duration": r.duration, "pattern": r.pattern,
                         "why": f"{r.format}; '{r.pattern}' title; {r.duration} long; {ratio:.1f}x this creator's usual pace",
                         "tags": ", ".join(r.tags[:6]) if isinstance(r.tags, list) else ""})
    if progress:
        progress(1.0, "Done")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("pace_ratio", ascending=False).reset_index(drop=True)
    return out, warnings


def high_potential(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(["Breakout", "Above usual pace"])].reset_index(drop=True) if len(df) else df


def learnings(df: pd.DataFrame, top_n: int = 10) -> dict:
    """What the fastest videos have in common (descriptive, small samples)."""
    if df.empty:
        return {"available": False}
    top = df.head(top_n)
    terms = [t for t, _ in top_terms(top["title"], 8)]
    return {"available": True, "n": len(top), "short_form_share": float((top["format"] == "Short-form (<=3 min)").mean()),
            "median_duration": format_duration(float(top["duration_seconds"].median())) if "duration_seconds" in top else "",
            "common_patterns": dict(Counter(top["pattern"]).most_common(3)), "common_terms": terms,
            "note": "Descriptive only: shows what fast-moving videos share right now, not what causes views."}


def trend_terms(*frames: pd.DataFrame, n: int = 12) -> list[str]:
    """Most common title/tag words across trending frames, for use in the title/hashtag generator."""
    texts: list[str] = []
    for f in frames:
        if f is None or len(f) == 0:
            continue
        texts += f["title"].tolist()
        if "tags" in f:
            texts += [", ".join(t) if isinstance(t, list) else str(t) for t in f["tags"]]
    return [t for t, _ in top_terms(texts, n)]
