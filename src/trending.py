"""Trending and high-potential video discovery.

Sources: YouTube's official Data API only (its own 'most popular' chart, search, and public channel uploads). TikTok, Instagram, X and other
platforms offer no public trending API and forbid scraping in their terms, so this tool does not scrape them. For other platforms the
creator can import a CSV from their own analytics export or from a tool they are licensed to use (see `import_external`).

'High potential' = a video moving fast for its age and/or far above its channel's usual audience, which hints that the topic or packaging
resonates beyond existing fans. It is a signal to study and adapt (topic, format, angle), never something to copy: copying titles, thumbnails
or footage risks copyright and reused-content problems.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.audience_profile import SHORT_FMT
from src.title_analysis import classify_pattern, wording_flags
from utils.data_processing import videos_to_df
from utils.validation import parse_channel_input

MAX_EXTERNAL_ROWS = 2000
TEMPLATE_CSV = "platform,creator,title,url,views,likes,comments,followers,posted_at\nTikTok,example_creator,My example video,https://example.com/v/1,125000,9000,310,48000,2026-10-20\n"
OUT_COLS = ["potential_score", "signal", "platform", "creator", "title", "views", "views_per_day", "views_per_follower", "engagement_rate", "followers", "age_days",
            "format", "pattern", "what_to_study", "published_at", "url", "video_id"]


def _now(now=None) -> pd.Timestamp:
    t = pd.Timestamp(now or datetime.now(timezone.utc))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) if len(s) > 1 else pd.Series([0.5] * len(s), index=s.index)


def _finish(d: pd.DataFrame, user_followers: float | None) -> pd.DataFrame:
    d = d.copy()
    d["age_days"] = ((_now() - pd.to_datetime(d["published_at"], utc=True)).dt.total_seconds() / 86400).clip(lower=1)
    d["views_per_day"] = d["views"] / d["age_days"]
    fol = pd.to_numeric(d["followers"], errors="coerce")
    d["views_per_follower"] = d["views"] / fol.where(fol >= 1000, np.nan)
    d["engagement_rate"] = (pd.to_numeric(d["likes"], errors="coerce").fillna(0) + pd.to_numeric(d["comments"], errors="coerce").fillna(0)) / d["views"].replace(0, np.nan)
    d["potential_score"] = (100 * (0.4 * _pct(d["views_per_day"]).fillna(0.5) + 0.4 * _pct(d["views_per_follower"].clip(upper=20)).fillna(0.5) +
                                   0.2 * _pct(d["engagement_rate"]).fillna(0.5))).round(0)
    d["format"] = np.where(d["duration_seconds"].fillna(999) <= 180, SHORT_FMT, "Long-form (>3 min)")
    d["pattern"] = d["title"].astype(str).map(classify_pattern)
    peer = (fol >= (user_followers or 0) * 0.2) & (fol <= (user_followers or 0) * 5) if user_followers else pd.Series(False, index=d.index)
    sig = np.where(d["views_per_follower"] >= 3, "Breakout: views far above its audience size", np.where(d["views_per_day"] >= d["views_per_day"].quantile(0.9), "Fast mover", "Steady"))
    d["signal"] = np.where(peer & (d["views_per_follower"] >= 1.5), "Breakout from a creator your size", sig)
    d["what_to_study"] = [f"{p}; {len(str(t))} chars; {'number, ' if any(c.isdigit() for c in str(t)) else ''}{f.lower()}; study the topic, hook and structure, not the exact title/thumbnail"
                          for p, t, f in zip(d["pattern"], d["title"], d["format"].str.split(" ").str[0])]
    return d.sort_values("potential_score", ascending=False).reset_index(drop=True)


def _frame(videos: list[dict], channels: list[dict]) -> pd.DataFrame:
    df = videos_to_df(videos)
    if df.empty:
        return df
    subs = {}
    for c in channels:
        subs[c.get("channel_id")] = c.get("subscribers")
        if c.get("_query_id"):
            subs[c["_query_id"]] = c.get("subscribers")
    df["followers"] = df["channel_id"].map(subs)
    df["platform"], df["creator"] = "YouTube", df["channel_title"]
    return df


def _enrich(client, videos: list[dict]) -> pd.DataFrame:
    ids = list({v["channel_id"] for v in videos if v.get("channel_id")})
    return _frame(videos, client.get_channels_by_ids(ids) if ids else [])


def trending(client, region: str = "US", category_id: str | None = None, user_followers: float | None = None) -> pd.DataFrame:
    """YouTube's most-popular chart for a region (about 3 quota units)."""
    vids = client.trending_videos(region, category_id, 50)
    df = _enrich(client, vids)
    return _finish(df, user_followers)[OUT_COLS] if not df.empty else df


def niche_outliers(client, keywords: list[str], days: int = 30, region: str | None = None, user_followers: float | None = None,
                   short_only: bool = False, max_results: int = 25) -> pd.DataFrame:
    """Recent videos for the creator's own keywords, ranked by speed and by views vs audience size (100 quota units per keyword)."""
    after = datetime.now(timezone.utc) - timedelta(days=days)
    seen: dict[str, dict] = {}
    for kw in [k for k in keywords if k.strip()][:3]:
        for v in client.search_videos(kw, after, "viewCount", region, max_results, short_only):
            seen.setdefault(v["video_id"], v)
    df = _enrich(client, list(seen.values()))
    return _finish(df, user_followers)[OUT_COLS] if not df.empty else df


def creator_outliers(client, links: list[str], sample_size: int = 30, min_multiple: float = 2.5) -> tuple[pd.DataFrame, list[str]]:
    """Latest videos of specific creators; 'outlier_x' = views vs that creator's own typical (median) video."""
    frames, problems = [], []
    for link in [l for l in links if l.strip()][:8]:
        try:
            b = client.fetch_channel_bundle(parse_channel_input(link), sample_size)
        except Exception as exc:  # noqa: BLE001 - one bad link must not stop the rest
            problems.append(f"{link.strip()}: {exc}")
            continue
        df = videos_to_df(b["videos"])
        if df.empty:
            problems.append(f"{link.strip()}: no public videos found")
            continue
        df["followers"], df["platform"], df["creator"] = b["channel"].get("subscribers"), "YouTube", b["channel"]["title"]
        age = (_now() - pd.to_datetime(df["published_at"], utc=True)).dt.days
        base = df[age >= 7]["views"].median() if (age >= 7).sum() >= 3 else df["views"].median()
        df["outlier_x"] = df["views"] / max(float(base), 1.0)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), problems
    d = _finish(pd.concat(frames, ignore_index=True), None)
    d["signal"] = np.where(d["outlier_x"] >= min_multiple, f"Outlier: {min_multiple:g}x+ this creator's usual views", d["signal"])
    cols = ["outlier_x"] + OUT_COLS
    return d.sort_values(["outlier_x", "views_per_day"], ascending=False).reset_index(drop=True)[cols], problems


def import_external(raw: pd.DataFrame, user_followers: float | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Score videos from any platform using a CSV the creator provides. Required: title, views, posted_at."""
    raw = raw.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in ("title", "views", "posted_at") if c not in raw]
    if missing:
        return pd.DataFrame(), [f"Missing required column(s): {', '.join(missing)}. Download the template to see the format."]
    notes: list[str] = []
    if len(raw) > MAX_EXTERNAL_ROWS:
        raw, notes = raw.head(MAX_EXTERNAL_ROWS), [f"Only the first {MAX_EXTERNAL_ROWS} rows were used."]
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)  # noqa: E731
    txt = lambda c, default: raw[c].astype(str) if c in raw else pd.Series(default, index=raw.index)  # noqa: E731
    d = pd.DataFrame({"title": raw["title"].astype(str), "views": num("views"), "published_at": pd.to_datetime(raw["posted_at"], utc=True, errors="coerce"),
                      "likes": num("likes"), "comments": num("comments"), "followers": num("followers"), "platform": txt("platform", "Imported"),
                      "creator": txt("creator", ""), "url": txt("url", ""), "duration_seconds": num("duration_seconds")})
    bad = int(d[["views", "published_at"]].isna().any(axis=1).sum())
    d = d.dropna(subset=["views", "published_at"])
    d = d[d["views"] >= 0]
    if bad:
        notes.append(f"{bad} row(s) skipped (views or posted_at could not be read).")
    if d.empty:
        return pd.DataFrame(), notes + ["No usable rows."]
    d["video_id"] = [f"import{i}" for i in range(len(d))]
    return _finish(d, user_followers)[OUT_COLS], notes
