"""Side-by-side competitor comparison from public data. Raw views are never treated as proof of 'better'."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.channel_analysis import upload_frequency
from src.title_analysis import classify_pattern
from src.topic_analysis import OTHER, auto_topics

LIMITATIONS = [
    "Compared channels differ in age, audience size, niche, upload history and content mix; raw views alone do not show which is 'better'.",
    "Views per 1,000 subscribers is a rough size-adjusted indicator; subscriber counts may be hidden or rounded by YouTube.",
    "Only public data from the selected sample of recent uploads is used. Private analytics (watch time, CTR, retention, revenue) are not available.",
    "Topics are inferred from titles/tags/descriptions, so they are approximate.",
]


def summarize_channel_row(role: str, channel: dict, df: pd.DataFrame) -> dict:
    freq = upload_frequency(df)
    subs = channel.get("subscribers")
    med_views = float(df["views"].median()) if len(df) else np.nan
    topics = auto_topics(df, max_topics=3) if len(df) >= 4 else pd.Series(dtype="object")
    main_topics = [t for t in topics.value_counts().index if t != OTHER][:3] if len(topics) else []
    top = df.sort_values("views", ascending=False).iloc[0] if len(df) else None
    patterns = df["title"].apply(classify_pattern).value_counts(normalize=True) if len(df) else pd.Series(dtype=float)
    return {
        "role": role,
        "channel": channel.get("title"),
        "url": channel.get("url"),
        "subscribers": subs,
        "total_views": channel.get("total_views"),
        "total_videos": channel.get("total_videos"),
        "channel_age_years": _age_years(channel.get("created_at")),
        "videos_in_sample": len(df),
        "uploads_per_30_days": freq["uploads_per_30_days"],
        "median_views": med_views,
        "median_views_per_day": float(df["views_per_day"].median()) if len(df) else np.nan,
        "median_views_per_1k_subs": (med_views / subs * 1000) if subs and not np.isnan(med_views) else np.nan,
        "short_form_share": float((df["format"] == "Short-form (<=3 min)").mean()) if len(df) else np.nan,
        "median_title_chars": float(df["title"].str.len().median()) if len(df) else np.nan,
        "main_topics": ", ".join(main_topics),
        "dominant_title_pattern": patterns.index[0] if len(patterns) else "",
        "top_video": top["title"] if top is not None else "",
        "top_video_url": top["url"] if top is not None else "",
        "top_video_views": int(top["views"]) if top is not None else None,
    }


def _age_years(created_at: str | None) -> float:
    ts = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.isna(ts):
        return np.nan
    return round((pd.Timestamp.now(tz="UTC") - ts).days / 365.25, 1)


def compare_channels(own: tuple[dict, pd.DataFrame], competitors: dict[str, tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    rows = [summarize_channel_row("You", *own)]
    rows += [summarize_channel_row("Competitor", ch, df) for ch, df in competitors.values()]
    return pd.DataFrame(rows)


def top_competitor_videos(competitors: dict[str, tuple[dict, pd.DataFrame]], n: int = 5) -> pd.DataFrame:
    frames = []
    for name, (_, df) in competitors.items():
        if len(df):
            top = df.sort_values("views", ascending=False).head(n)[["channel_title", "title", "url", "views", "published_at", "relative_to_median"]]
            frames.append(top)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def strategy_differences(table: pd.DataFrame) -> list[str]:
    """Plain-language differences between you and each competitor (descriptive, not judgemental)."""
    if len(table) < 2:
        return []
    me = table.iloc[0]
    notes: list[str] = []
    for _, c in table.iloc[1:].iterrows():
        name = c["channel"]
        if _ok(me["uploads_per_30_days"]) and _ok(c["uploads_per_30_days"]) and me["uploads_per_30_days"] > 0:
            ratio = c["uploads_per_30_days"] / me["uploads_per_30_days"]
            if ratio >= 1.5 or ratio <= 0.67:
                notes.append(f"{name} uploads about {ratio:.1f}x as often as you in the sampled period.")
        if _ok(me["short_form_share"]) and _ok(c["short_form_share"]) and abs(me["short_form_share"] - c["short_form_share"]) >= 0.3:
            notes.append(f"{name} publishes {c['short_form_share']:.0%} short-form vs your {me['short_form_share']:.0%}.")
        if c["dominant_title_pattern"] and c["dominant_title_pattern"] != me["dominant_title_pattern"]:
            notes.append(f"{name} leans on '{c['dominant_title_pattern']}' titles; yours lean on '{me['dominant_title_pattern']}'.")
        if c["main_topics"]:
            notes.append(f"{name}'s main inferred topics: {c['main_topics']}.")
    return notes


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))
