"""Static matplotlib charts (PNG bytes) for the PDF and Word reports. No browser or kaleido needed."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT, GREY = "#d62828", "#8d99ae"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _shorten(s: str, n: int = 38) -> str:
    return s if len(s) <= n else s[: n - 1] + "..."


def views_timeline(df: pd.DataFrame) -> bytes | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    d = df.sort_values("published_at")
    ax.bar(d["published_at"].dt.tz_convert("UTC").dt.tz_localize(None), d["views"], width=max(1.0, 0.6 * (d["published_at"].max() - d["published_at"].min()).days / max(len(d), 1)), color=ACCENT)
    ax.axhline(d["views"].median(), color=GREY, linestyle="--", linewidth=1, label="Sample median")
    ax.set_title("Public views by upload date (selected sample)", fontsize=10)
    ax.set_ylabel("Views")
    ax.legend(fontsize=8)
    return _png(fig)


def top_videos_bar(df: pd.DataFrame, n: int = 8) -> bytes | None:
    if df.empty:
        return None
    d = df.sort_values("views", ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([_shorten(t) for t in d["title"]], d["views"], color=ACCENT)
    ax.set_title("Most-viewed videos in the sample", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    return _png(fig)


def topic_bar(topics: pd.DataFrame) -> bytes | None:
    if topics is None or topics.empty:
        return None
    d = topics.sort_values("median_views").tail(10)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(d["topic"], d["median_views"], color=ACCENT)
    ax.set_title("Median views by topic (inferred from metadata)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return _png(fig)


def competitor_bar(comp: pd.DataFrame) -> bytes | None:
    if comp is None or comp.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    names = [_shorten(str(n), 16) for n in comp["channel"]]
    axes[0].bar(names, comp["median_views"], color=[ACCENT if r == "You" else GREY for r in comp["role"]])
    axes[0].set_title("Median views (sample)", fontsize=9)
    axes[1].bar(names, comp["uploads_per_30_days"].fillna(0), color=[ACCENT if r == "You" else GREY for r in comp["role"]])
    axes[1].set_title("Uploads per 30 days", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    return _png(fig)


def gap_bar(gaps: pd.DataFrame, n: int = 8) -> bytes | None:
    if gaps is None or gaps.empty:
        return None
    d = gaps.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([_shorten(t, 34) for t in d["topic"]], d["priority_score"], color=ACCENT)
    ax.set_xlim(0, 100)
    ax.set_title("Content-gap priority score (0-100, see methodology)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return _png(fig)


def build_charts(results: dict) -> dict[str, bytes]:
    charts = {
        "Views timeline": views_timeline(results.get("videos", pd.DataFrame())),
        "Top videos": top_videos_bar(results.get("videos", pd.DataFrame())),
        "Topics": topic_bar(results.get("topics")),
        "Competitors": competitor_bar(results.get("competitors")),
        "Content gaps": gap_bar(results.get("gaps")),
    }
    return {k: v for k, v in charts.items() if v}
