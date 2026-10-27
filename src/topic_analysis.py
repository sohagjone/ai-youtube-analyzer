"""Keyword-based topic grouping. Works from metadata (title, tags, start of description), not full video content."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from utils.data_processing import term_set

OTHER = "Other / unclassified"


def video_text(row: pd.Series, include_description: bool = True) -> str:
    desc = re.sub(r"https?://\S+", " ", str(row.get("description", "")))[:160] if include_description else ""
    tags = " , ".join(row["tags"]) if isinstance(row.get("tags"), (list, tuple)) else ""
    return f"{row.get('title', '')} . {tags} . {desc}"


def _topic_docs(df: pd.DataFrame) -> list[set[str]]:
    """Term sets per video. Description terms are used only when they are not channel boilerplate
    (terms present in >40% of descriptions are dropped; tiny samples ignore descriptions)."""
    n = len(df)
    title_tag = [term_set(f"{r.title} . {' , '.join(r.tags) if isinstance(r.tags, list) else ''}") for r in df.itertuples()]
    if n < 5:
        return title_tag
    desc = [term_set(re.sub(r"https?://\S+", " ", str(d))[:160]) for d in df["description"]]
    df_count: Counter[str] = Counter()
    for terms in desc:
        df_count.update(terms)
    return [tt | {t for t in d if df_count[t] <= 0.4 * n} for tt, d in zip(title_tag, desc)]


def auto_topics(df: pd.DataFrame, max_topics: int = 8, min_videos: int = 2) -> pd.Series:
    """Greedy keyword clustering: repeatedly pick the term covering most unassigned videos.
    Terms present in >60% of videos are too generic to be a topic and are skipped (samples of 8+)."""
    if df.empty:
        return pd.Series(dtype="object")
    docs = _topic_docs(df)
    n = len(docs)
    overall: Counter[str] = Counter()
    for terms in docs:
        overall.update(terms)
    too_generic = {t for t, c in overall.items() if n >= 8 and c > 0.6 * n}
    assigned: dict[int, str] = {}
    for _ in range(max_topics):
        counts: Counter[str] = Counter()
        for i, terms in enumerate(docs):
            if i not in assigned:
                counts.update(terms - too_generic)
        best, best_score = None, 0.0
        for term, c in counts.items():
            if c < min_videos:
                continue
            score = c * (1.25 if " " in term else 1.0)
            if score > best_score or (score == best_score and best and len(term) > len(best)):
                best, best_score = term, score
        if best is None:
            break
        for i, terms in enumerate(docs):
            if i not in assigned and best in terms:
                assigned[i] = best.title()
    return pd.Series([assigned.get(i, OTHER) for i in range(len(df))], index=df.index, name="topic")


def parse_topic_mapping(text: str) -> dict[str, list[str]]:
    """Parse lines like 'Espresso: espresso, shot, crema' into a mapping."""
    mapping: dict[str, list[str]] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        name, kws = line.split(":", 1)
        words = [k.strip().lower() for k in kws.split(",") if k.strip()]
        if name.strip() and words:
            mapping[name.strip()] = words
    return mapping


def custom_topics(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.Series:
    """Assign each video to the user-defined topic with the most keyword hits (ties: first listed)."""
    labels = []
    for _, r in df.iterrows():
        text = video_text(r).lower()
        best, best_hits = OTHER, 0
        for topic, kws in mapping.items():
            hits = sum(1 for k in kws if re.search(r"\b" + re.escape(k) + r"\b", text))
            if hits > best_hits:
                best, best_hits = topic, hits
        labels.append(best)
    return pd.Series(labels, index=df.index, name="topic")


def assign_topics(df: pd.DataFrame, mapping: dict[str, list[str]] | None = None, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["topic"] = custom_topics(out, mapping) if mapping else auto_topics(out, **kwargs)
    return out


def topic_summary(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Per-topic counts and performance with cautious observations."""
    if df.empty or "topic" not in df:
        return pd.DataFrame(columns=["topic", "videos", "share", "total_views", "median_views", "median_relative",
                                     "spread_ratio", "recent_share", "older_share", "observation"])
    cutoff = df["published_at"].median()
    recent = df["published_at"] >= cutoff
    rows = []
    for topic, sub in df.groupby("topic"):
        n = len(sub)
        med = float(sub["views"].median())
        q1, q3 = sub["views"].quantile(0.25), sub["views"].quantile(0.75)
        spread = float((q3 - q1) / med) if med > 0 else np.nan
        rel = float(sub["relative_to_median"].median()) if "relative_to_median" in sub else np.nan
        share = n / len(df)
        if topic == OTHER:
            obs = "Unclassified videos"
        elif n < min_n:
            obs = "Insufficient data (fewer than 3 videos)"
        elif not np.isnan(spread) and spread >= 1.5:
            obs = "Inconsistent performance"
        elif rel >= 1.25:
            obs = "Strong observed performance"
        elif rel <= 0.75:
            obs = "Weaker observed performance"
        else:
            obs = "Typical performance"
        if topic != OTHER and n >= 2 and share < 0.15 and rel >= 1.1:
            obs += "; underrepresented relative to its performance"
        rows.append({
            "topic": topic, "videos": n, "share": share, "total_views": int(sub["views"].sum()),
            "median_views": med, "median_relative": rel, "spread_ratio": spread,
            "recent_share": float(recent[sub.index].mean()), "older_share": float((~recent[sub.index]).mean()),
            "observation": obs,
        })
    res = pd.DataFrame(rows).sort_values("videos", ascending=False).reset_index(drop=True)
    # Recent vs older: share of each topic's videos published in the newer half of the sample
    res["recent_vs_older"] = np.where(res["recent_share"] > 0.65, "Mostly recent", np.where(res["older_share"] > 0.65, "Mostly older", "Balanced"))
    return res


def recent_vs_older(df: pd.DataFrame) -> pd.DataFrame:
    """Topic mix in the newer half vs the older half of the sample."""
    if df.empty or "topic" not in df:
        return pd.DataFrame()
    cutoff = df["published_at"].median()
    new, old = df[df["published_at"] >= cutoff], df[df["published_at"] < cutoff]
    a = new["topic"].value_counts(normalize=True).rename("recent_share")
    b = old["topic"].value_counts(normalize=True).rename("older_share")
    return pd.concat([a, b], axis=1).fillna(0).reset_index(names="topic").assign(
        change=lambda d: d["recent_share"] - d["older_share"]).sort_values("change", ascending=False)
