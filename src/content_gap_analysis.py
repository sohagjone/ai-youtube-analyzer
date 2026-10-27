"""Content-gap finder with a transparent priority score.

A *content gap* (a topic competitors cover that you rarely do) is NOT the same as a proven *search-demand
opportunity*. No search-volume data is used; "demand evidence" below means how competitors' videos on the topic
performed relative to their own channel medians, and how many such videos exist.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.data_processing import term_set, tokenize

WEIGHTS = {"relevance": 0.30, "demand_evidence": 0.30, "differentiation": 0.20, "feasibility": 0.20}
BEGINNER_TERMS = {"beginner", "beginners", "basics", "intro", "introduction", "101", "starter", "getting started", "explained", "simple"}
ADVANCED_TERMS = {"advanced", "masterclass", "deep dive", "pro", "expert", "in-depth", "professional"}
QUESTION_STARTS = ("how", "why", "what", "when", "which", "can", "should", "is", "are", "does")

DISCLAIMER = ("Content gap = a topic your channel rarely covers; it is not proof of audience or search demand. "
              "No search-volume data was used.")

GAP_COLUMNS = ["rank", "gap_type", "topic", "audience", "evidence", "competing_videos", "why_test", "suggested_title",
               "format", "outline", "priority_score", "confidence", "limitations", "relevance", "demand_evidence",
               "differentiation", "feasibility"]


def priority_score(relevance: float, demand: float, differentiation: float, feasibility: float) -> float:
    s = (WEIGHTS["relevance"] * relevance + WEIGHTS["demand_evidence"] * demand
         + WEIGHTS["differentiation"] * differentiation + WEIGHTS["feasibility"] * feasibility)
    return round(100 * float(np.clip(s, 0, 1)), 1)


def _confidence(videos: int, channels: int) -> str:
    if videos >= 4 and channels >= 2:
        return "High"
    if videos >= 3 or channels >= 2:
        return "Medium"
    return "Low"


def _outline(topic: str, fmt: str) -> str:
    if fmt.startswith("Short"):
        return (f"0-3s hook on {topic} | one clear point or demo | one concrete example from your experience | "
                "CTA (follow / watch the long-form version)")
    return (f"Hook: why {topic} matters to your viewers | Core explanation covering the angles competitors use | "
            "Your unique angle or experience (what they don't show) | Common mistakes or FAQs | Recap and next-video CTA")


def _suggest_title(topic: str, level: str, fmt: str) -> str:
    t = topic.title()
    if level == "Beginner":
        return f"{t} for Beginners: Where to Start"
    if level == "Advanced":
        return f"{t}: An Advanced Walkthrough"
    if fmt.startswith("Short"):
        return f"{t} in Under a Minute"
    return f"{t}: What to Know Before You Start"


def _dominant_format(df: pd.DataFrame) -> str:
    if df.empty or "format" not in df:
        return "Long-form (>3 min)"
    return df["format"].value_counts().index[0]


def _own_vocab(own: pd.DataFrame, n: int = 30) -> set[str]:
    from collections import Counter
    c: Counter[str] = Counter()
    for t in own["title"]:
        c.update(set(tokenize(t)))
    return {k for k, _ in c.most_common(n)}


def _links(rows: pd.DataFrame, limit: int = 3) -> str:
    top = rows.sort_values("views", ascending=False).head(limit)
    return "\n".join(f"{r.title} ({r.channel_title}, {int(r.views):,} views) {r.url}" for r in top.itertuples())


def _competitor_gaps(own: pd.DataFrame, comp: pd.DataFrame, n_channels: int) -> list[dict]:
    if own.empty or comp.empty:
        return []
    own_docs = [term_set(t) for t in own["title"]]
    own_count = defaultdict(int)
    for terms in own_docs:
        for t in terms:
            own_count[t] += 1
    comp_docs = [term_set(f"{r.title} {' '.join(r.tags) if isinstance(r.tags, list) else ''}") for r in comp.itertuples()]
    term_rows: dict[str, list[int]] = defaultdict(list)
    for i, terms in enumerate(comp_docs):
        for t in terms:
            term_rows[t].append(i)
    vocab = _own_vocab(own)
    own_dom = _dominant_format(own)
    candidates = []
    for term, idxs in term_rows.items():
        if len(idxs) < 2 or own_count.get(term, 0) > max(1, 0.05 * len(own)):
            continue
        sub = comp.iloc[idxs]
        channels = sub["channel_title"].nunique()
        relevance = float(np.mean([bool(vocab & set(tokenize(t))) for t in sub["title"]]))
        med_rel = float(sub["relative_to_median"].median()) if "relative_to_median" in sub else 1.0
        demand = 0.6 * min(1.0, med_rel / 2) + 0.4 * min(1.0, len(sub) / 6)
        own_c = own_count.get(term, 0)
        diff = (1.0 if own_c == 0 else 0.5) * (0.6 + 0.4 * (1 - channels / max(n_channels, 1)))
        feas = 0.3 + 0.7 * float((sub["format"] == own_dom).mean())
        candidates.append((priority_score(relevance, demand, diff, feas), term, idxs, relevance, demand, diff, feas, channels, med_rel))
    candidates.sort(key=lambda c: (-c[0], -len(c[2]), -len(c[1])))
    chosen: list[tuple] = []
    for cand in candidates:
        s = set(cand[2])
        if any(len(s & set(c[2])) / len(s | set(c[2])) >= 0.7 for c in chosen):
            continue  # near-duplicate of a better-scored term
        chosen.append(cand)
        if len(chosen) >= 12:
            break
    rows = []
    for score, term, idxs, rel, dem, diff, feas, channels, med_rel in chosen:
        sub = comp.iloc[idxs]
        fmt = sub["format"].value_counts().index[0]
        qs = [t for t in sub["title"] if t.strip().lower().split(" ")[0] in QUESTION_STARTS or "?" in t][:2]
        evidence = (f"{len(sub)} competitor video(s) across {channels} channel(s) mention '{term}'; median performance "
                    f"{med_rel:.2f}x their channel medians. You have {own_count.get(term, 0)} video(s) on it.")
        if qs:
            evidence += " Question-style titles: " + " | ".join(qs)
        rows.append(_row("Competitor topic gap", term.title(), "Your existing audience (assumed - verify)", evidence,
                         _links(sub), "Competitors repeatedly cover this and you rarely do; worth a test video to see if your audience responds.",
                         fmt, score, rel, dem, diff, feas, _confidence(len(sub), channels),
                         "Competitor view counts depend on their audience size and age; demand not confirmed by search data."))
    return rows


def _row(gap_type, topic, audience, evidence, competing, why, fmt, score, rel, dem, diff, feas, conf, limitation, level="General", title=None):
    return {
        "gap_type": gap_type, "topic": topic, "audience": audience, "evidence": evidence, "competing_videos": competing,
        "why_test": why, "suggested_title": title or _suggest_title(topic, level, fmt), "format": fmt,
        "outline": _outline(topic.lower(), fmt), "priority_score": score, "confidence": conf,
        "limitations": limitation + " " + DISCLAIMER, "relevance": round(rel, 2), "demand_evidence": round(dem, 2),
        "differentiation": round(diff, 2), "feasibility": round(feas, 2),
    }


def _own_followups(own: pd.DataFrame, topic_summary: pd.DataFrame | None) -> list[dict]:
    rows = []
    if topic_summary is None or topic_summary.empty:
        return rows
    fmt = _dominant_format(own)
    for r in topic_summary.itertuples():
        if r.topic.startswith("Other") or r.videos < 2 or not (r.median_relative >= 1.25):
            continue
        dem = min(1.0, r.median_relative / 2)
        conf = "Medium" if r.videos >= 3 else "Low"
        rows.append(_row("Follow-up / series on a strong topic", f"{r.topic} (follow-up)", "Existing viewers of these videos",
                         f"Your {r.videos} '{r.topic}' videos have a median of {r.median_relative:.2f}x your channel median in this sample.",
                         "", "A proven-for-you topic: a follow-up or series tests whether the interest repeats.", fmt,
                         priority_score(1.0, dem, 0.5, 1.0), 1.0, dem, 0.5, 1.0, conf,
                         "Based on your own sample only; older videos had longer to accumulate views.",
                         title=f"{r.topic}: Part 2 - What's Next"))
    return rows[:4]


def _outdated(own: pd.DataFrame) -> list[dict]:
    if own.empty or "age_days" not in own:
        return []
    old = own[own["age_days"] > 730]
    if old.empty:
        return []
    cutoff = own["views_per_day"].median()
    cand = old[old["views_per_day"] >= cutoff].sort_values("views_per_day", ascending=False).head(2)
    rows = []
    for r in cand.itertuples():
        rows.append(_row("Update outdated content", f"Refresh: {r.title}", "Viewers still finding the older video",
                         f"Published {int(r.age_days / 365.25)}+ years ago and still pace of {r.views_per_day:,.1f} views/day, at or above your median.",
                         r.url, "Older videos that still attract views may contain outdated information; an updated version could serve them.",
                         r.format, priority_score(1.0, 0.6, 0.5, 0.9), 1.0, 0.6, 0.5, 0.9, "Medium",
                         "Check the content is actually outdated before remaking it.", title=f"{r.title} (Updated)"))
    return rows


def _level_gaps(own: pd.DataFrame, comp: pd.DataFrame) -> list[dict]:
    rows = []
    own_text = " ".join(own["title"]).lower() if len(own) else ""
    comp_text = [t.lower() for t in comp["title"]] if len(comp) else []
    fmt = _dominant_format(own)
    for level, terms in (("Beginner", BEGINNER_TERMS), ("Advanced", ADVANCED_TERMS)):
        own_has = any(re.search(r"\b" + re.escape(t) + r"\b", own_text) for t in terms)
        comp_hits = sum(any(re.search(r"\b" + re.escape(t) + r"\b", c) for t in terms) for c in comp_text)
        if not own_has and comp_hits >= 2:
            rows.append(_row(f"{level} content gap", f"{level}-level introduction to your main topic", f"{level} viewers",
                             f"None of your sampled titles signal {level.lower()}-level content; {comp_hits} competitor titles do.",
                             "", f"Competitors serve {level.lower()} viewers and your titles don't signal it.", fmt,
                             priority_score(0.9, min(1.0, comp_hits / 6), 0.8, 0.8), 0.9, min(1.0, comp_hits / 6), 0.8, 0.8,
                             "Medium" if comp_hits >= 3 else "Low",
                             "Title keywords are only a rough proxy for content level.", level=level))
    return rows


def find_content_gaps(own: pd.DataFrame, competitors: dict[str, pd.DataFrame] | None = None,
                      topic_summary: pd.DataFrame | None = None, max_results: int = 15) -> pd.DataFrame:
    """Return ranked opportunities. Works without competitors (own-channel ideas only)."""
    competitors = competitors or {}
    comp = pd.concat([d for d in competitors.values() if len(d)], ignore_index=True) if any(len(d) for d in competitors.values()) else pd.DataFrame(columns=own.columns)
    rows: list[dict] = []
    if len(own):
        rows += _competitor_gaps(own, comp, max(len(competitors), 1))
        rows += _own_followups(own, topic_summary)
        rows += _outdated(own)
        rows += _level_gaps(own, comp)
    if not rows:
        return pd.DataFrame(columns=GAP_COLUMNS)
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).head(max_results).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df[GAP_COLUMNS]


def refine_gaps_with_ai(provider, gaps: pd.DataFrame, niche: str = "", top_n: int = 5) -> pd.DataFrame:
    """Ask the AI provider for better titles/outlines for the top gaps; keeps rule-based text on any failure."""
    out = gaps.copy()
    if out.empty:
        return out
    items = [{"i": int(i), "topic": r.topic, "evidence": r.evidence[:300], "format": r.format}
             for i, r in out.head(top_n).iterrows()]
    system = ("You are a careful YouTube strategist. Use ONLY the evidence given; do not invent statistics. "
              "No clickbait or promises of results. Return ONLY JSON.")
    prompt = (f"Niche: {niche or 'unspecified'}\nIdeas: {items}\nReturn a JSON list of objects: "
              '[{"i": <same index>, "title": "...", "outline": "5 short beats separated by |"}]')
    try:
        data = provider.complete_json(system, prompt, max_tokens=1200)
    except Exception:  # noqa: BLE001 - AI is optional; keep rule-based output
        return out
    for d in data if isinstance(data, list) else []:
        try:
            i = int(d["i"])
            if i in out.index and d.get("title"):
                out.at[i, "suggested_title"] = str(d["title"]).strip()
                if d.get("outline"):
                    out.at[i, "outline"] = str(d["outline"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
    return out
