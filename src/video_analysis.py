"""Video-level metrics. Every number here is derived from PUBLIC data only."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

SHORT_MAX_SECONDS = 180  # YouTube Shorts can be up to 3 minutes; the API has no explicit Shorts flag
TOO_NEW_DAYS = 30


def _now(now: datetime | pd.Timestamp | None) -> pd.Timestamp:
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def filter_by_date(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Keep videos published within [start, end]; naive datetimes are treated as UTC."""
    out = df
    if start is not None:
        out = out[out["published_at"] >= _to_utc(start)]
    if end is not None:
        out = out[out["published_at"] <= _to_utc(end)]
    return out.reset_index(drop=True)


def enrich_videos(df: pd.DataFrame, now: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Add age, views/day, performance relative to the sample median, engagement and format columns.

    Definitions (also shown in the app's methodology panel):
    * age_days            = days since publication, floored at 1 to avoid division by zero.
    * views_per_day       = views / age_days  (a crude pace metric; older videos accumulate views over longer periods).
    * relative_to_median  = views / median(views in this sample).
    * engagement_rate     = (likes + comments) / views using whichever counts are public; NaN when neither is.
    """
    out = df.copy()
    if out.empty:
        for c in ("age_days", "views_per_day", "relative_to_median", "like_rate", "comment_rate",
                  "engagement_rate", "format", "duration_bucket", "performance_flag"):
            out[c] = pd.Series(dtype="object")
        return out
    now_ts = _now(now)
    out["age_days"] = ((now_ts - out["published_at"]).dt.total_seconds() / 86400).clip(lower=1)
    out["views_per_day"] = out["views"] / out["age_days"]
    med = float(out["views"].median())
    out["relative_to_median"] = out["views"] / med if med > 0 else np.nan
    views = out["views"].replace(0, np.nan)
    out["like_rate"] = out["likes"] / views
    out["comment_rate"] = out["comments"] / views
    both_missing = out["likes"].isna() & out["comments"].isna()
    out["engagement_rate"] = (out["likes"].fillna(0) + out["comments"].fillna(0)) / views
    out.loc[both_missing, "engagement_rate"] = np.nan
    out["format"] = np.where(out["duration_seconds"] <= SHORT_MAX_SECONDS, "Short-form (<=3 min)", "Long-form (>3 min)")
    out["duration_bucket"] = pd.cut(
        out["duration_seconds"], bins=[-1, 60, 180, 600, 1200, float("inf")],
        labels=["<=1 min", "1-3 min", "3-10 min", "10-20 min", "20+ min"]).astype(str)
    out["performance_flag"] = out.apply(_flag, axis=1)
    return out


def _flag(row: pd.Series) -> str:
    rel = row.get("relative_to_median")
    if pd.isna(rel):
        return "Typical"
    if rel >= 1.5:
        return "Above typical"
    if rel <= 0.5:
        return "Too new to judge" if row["age_days"] < TOO_NEW_DAYS else "Below typical"
    return "Typical"


def summarize_videos(df: pd.DataFrame) -> dict:
    """Descriptive statistics for the selected sample (NOT lifetime channel statistics)."""
    if df.empty:
        return {"videos_in_sample": 0}
    v = df["views"]
    return {
        "videos_in_sample": int(len(df)),
        "sample_start": df["published_at"].min().isoformat(),
        "sample_end": df["published_at"].max().isoformat(),
        "total_views_in_sample": int(v.sum()),
        "mean_views": float(v.mean()),
        "median_views": float(v.median()),
        "p25_views": float(v.quantile(0.25)),
        "p75_views": float(v.quantile(0.75)),
        "min_views": int(v.min()),
        "max_views": int(v.max()),
        "median_views_per_day": float(df["views_per_day"].median()),
        "mean_views_per_day": float(df["views_per_day"].mean()),
        "short_form_share": float((df["format"] == "Short-form (<=3 min)").mean()),
        "median_duration_seconds": float(df["duration_seconds"].median()),
        "median_engagement_rate": float(df["engagement_rate"].median()) if df["engagement_rate"].notna().any() else None,
        "videos_with_public_likes": int(df["likes"].notna().sum()),
        "videos_with_public_comments": int(df["comments"].notna().sum()),
    }


def top_videos(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.sort_values("views", ascending=False).head(n).reset_index(drop=True)


def lowest_videos(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Lowest-viewed videos in the sample. Very new uploads are included but flagged."""
    return df.sort_values("views", ascending=True).head(n).reset_index(drop=True)


def below_typical(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["performance_flag"] == "Below typical"].reset_index(drop=True)


def format_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("format").agg(videos=("video_id", "count"), median_views=("views", "median"),
                                 median_views_per_day=("views_per_day", "median"),
                                 median_engagement_rate=("engagement_rate", "median")).reset_index()
    return g
