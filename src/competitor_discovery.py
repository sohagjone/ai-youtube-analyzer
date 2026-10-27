"""Automatic competitor discovery: finds channels that publish recent videos on the same topics as yours.

Method (all public data, official API): build search queries from your own topics/keywords -> collect channels
whose recent videos match -> drop yourself and auto-generated '- Topic' channels -> score by how often a channel
appears across queries (topical match) and by audience-size similarity -> keep the best N.
This is a heuristic, not a ranking of who is a 'true' competitor.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.topic_analysis import OTHER
from utils.data_processing import top_terms
from utils.validation import ParsedChannel

MIN_VIDEOS = 5


def build_queries(own_df: pd.DataFrame, topic_summary: pd.DataFrame | None, max_queries: int = 4) -> list[str]:
    """Search phrases from the channel's strongest recurring topics, anchored by its core keyword."""
    if own_df.empty:
        return []
    tags = Counter(t.lower() for tags in own_df["tags"] for t in (tags or []) if isinstance(t, str))
    core = tags.most_common(1)[0][0] if tags else (top_terms(own_df["title"], 1) or [("", 0)])[0][0]
    queries: list[str] = []
    if topic_summary is not None and len(topic_summary):
        for topic in topic_summary.sort_values("videos", ascending=False)["topic"]:
            if topic == OTHER:
                continue
            t = str(topic).lower()
            queries.append(t if core and core in t else f"{t} {core}".strip())
    for kw, _ in top_terms(own_df["title"], 6):
        q = f"{kw} {core}".strip() if core and core != kw else kw
        queries.append(q)
    if core:
        queries.append(core)
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


def _size_similarity(own_subs, subs) -> float:
    if not own_subs or not subs or own_subs <= 0 or subs <= 0:
        return 0.5
    return float(1 - min(1.0, abs(math.log10(subs / own_subs)) / 2))


def discover_competitors(client, own_channel: dict, own_df: pd.DataFrame, topic_summary: pd.DataFrame | None,
                         n: int = 10, pool: int = 30, now: datetime | None = None) -> tuple[list[ParsedChannel], pd.DataFrame, list[str]]:
    """Return (channel refs, explanation table, notes). Raises YouTubeAPIError subclasses on API failure."""
    now = now or datetime.now(timezone.utc)
    notes: list[str] = []
    queries = build_queries(own_df, topic_summary)
    if not queries:
        return [], pd.DataFrame(), ["Not enough video metadata to build competitor search queries."]
    cands = client.find_channel_candidates(queries, 50, now - timedelta(days=365))
    own_id = own_channel.get("channel_id")
    cands = [c for c in cands if c["channel_id"] != own_id and not str(c.get("title", "")).endswith(" - Topic")][:pool]
    if not cands:
        return [], pd.DataFrame(), [f"No similar channels were found for the queries: {', '.join(queries)}."]
    details = client.get_channels_by_ids([c["channel_id"] for c in cands])
    by_key = {(d.get("_query_id") or d.get("channel_id")): d for d in details}
    max_hits = max(c["hits"] for c in cands)
    own_subs = own_channel.get("subscribers")
    rows = []
    for c in cands:
        d = by_key.get(c["channel_id"])
        if not d or not d.get("uploads_playlist") and not getattr(client, "is_sample", False):
            continue
        if (d.get("total_videos") or 0) < MIN_VIDEOS:
            continue
        sim = _size_similarity(own_subs, d.get("subscribers"))
        score = 0.6 * (c["hits"] / max_hits) + 0.4 * sim
        rows.append({"channel": d.get("title") or c["title"], "channel_id": c["channel_id"], "subscribers": d.get("subscribers"),
                     "query_hits": c["hits"], "size_similarity": round(sim, 2), "match_score": round(score, 3),
                     "url": d.get("url"), "why": f"Appeared in {c['hits']} of the recent-video results for: {', '.join(queries[:3])}"})
    table = pd.DataFrame(rows).sort_values("match_score", ascending=False).head(n).reset_index(drop=True) if rows else pd.DataFrame()
    if len(table) < n:
        notes.append(f"Only {len(table)} suitable competitor channels were found (target {n}); try again later or after publishing more videos.")
    notes.append("Competitors were chosen automatically from channels publishing recent videos on your topics (queries: " + "; ".join(queries) + ").")
    sample = getattr(client, "is_sample", False)
    refs = [ParsedChannel("handle", "@" + cid) if sample else ParsedChannel("id", cid) for cid in table.get("channel_id", [])]
    return refs, table, notes
