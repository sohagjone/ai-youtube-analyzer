"""Channel overview: lifetime statistics are kept separate from statistics of the selected sample."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.video_analysis import summarize_videos

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def upload_frequency(df: pd.DataFrame) -> dict:
    """Uploads per 30 days (from gaps between uploads in the sample) and gap statistics."""
    if len(df) < 2:
        return {"uploads_per_30_days": None, "median_gap_days": None, "days_since_last_upload": None}
    dates = df["published_at"].sort_values()
    span_days = max((dates.iloc[-1] - dates.iloc[0]).total_seconds() / 86400, 1.0)
    gaps = dates.diff().dropna().dt.total_seconds() / 86400
    now = pd.Timestamp(datetime.now(timezone.utc))
    return {
        "uploads_per_30_days": float((len(dates) - 1) / span_days * 30),
        "median_gap_days": float(gaps.median()),
        "days_since_last_upload": float((now - dates.iloc[-1]).total_seconds() / 86400),
    }


def publishing_patterns(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Weekday / hour (UTC) publishing patterns with median views of the sample."""
    if df.empty:
        return {"weekday": pd.DataFrame(), "hour": pd.DataFrame()}
    d = df.copy()
    d["weekday"] = d["published_at"].dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    d["hour_utc"] = d["published_at"].dt.hour
    wd = d.groupby("weekday").agg(videos=("video_id", "count"), median_views=("views", "median")).reindex(WEEKDAYS).fillna(0).reset_index()
    hr = d.groupby("hour_utc").agg(videos=("video_id", "count"), median_views=("views", "median")).reset_index()
    return {"weekday": wd, "hour": hr}


def summarize_channel(channel: dict, df: pd.DataFrame) -> dict:
    """Return {'lifetime': {...}, 'sample': {...}, 'upload_frequency': {...}}."""
    created = pd.to_datetime(channel.get("created_at"), utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    total_videos, total_views = channel.get("total_videos"), channel.get("total_views")
    lifetime = {
        "channel_name": channel.get("title"),
        "channel_url": channel.get("url"),
        "description": channel.get("description"),
        "created_at": None if pd.isna(created) else created.isoformat(),
        "channel_age_years": None if pd.isna(created) else round((now - created).days / 365.25, 2),
        "subscribers": channel.get("subscribers"),
        "subscribers_hidden": channel.get("subscribers_hidden", False),
        "total_views": total_views,
        "total_videos": total_videos,
        "lifetime_avg_views_per_video": (total_views / total_videos) if total_views is not None and total_videos else None,
        "country": channel.get("country"),
    }
    return {
        "lifetime": lifetime,
        "sample": summarize_videos(df),
        "upload_frequency": upload_frequency(df),
    }


def overview_table(summary: dict) -> pd.DataFrame:
    """Flat Metric / Value / Scope table used by exports."""
    rows: list[dict] = []
    life, samp, freq = summary["lifetime"], summary["sample"], summary["upload_frequency"]
    n = samp.get("videos_in_sample", 0)

    def add(metric, value, scope):
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            rows.append({"Metric": metric, "Value": value, "Scope": scope})

    life_scope = "Lifetime (whole channel, public)"
    add("Channel name", life["channel_name"], life_scope)
    add("Created", (life["created_at"] or "")[:10] or None, life_scope)
    add("Subscribers (public)", life["subscribers"] if not life["subscribers_hidden"] else "Hidden by creator", life_scope)
    add("Total views", life["total_views"], life_scope)
    add("Total videos", life["total_videos"], life_scope)
    add("Lifetime average views per video", life["lifetime_avg_views_per_video"], life_scope)
    scope = f"Selected sample ({n} videos)"
    add("Median views per video", samp.get("median_views"), scope)
    add("Mean views per video", samp.get("mean_views"), scope)
    add("Total views in sample", samp.get("total_views_in_sample"), scope)
    add("Median views per day since publication", samp.get("median_views_per_day"), scope)
    add("Short-form share", samp.get("short_form_share"), scope)
    add("Uploads per 30 days", freq.get("uploads_per_30_days"), scope)
    add("Median days between uploads", freq.get("median_gap_days"), scope)
    return pd.DataFrame(rows)
