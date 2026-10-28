"""Channel context ('what your audience has responded to') shared by the Video Studio features.

IMPORTANT: a channel's subscribers' private behaviour (retention, traffic sources, demographics) is NOT public. Everything
labelled 'audience behaviour' here is inferred from PUBLIC proxies: views, views/day, likes+comments, formats, durations and
publishing times of the channel's own videos. Treat it as observed patterns, not as proof of what viewers want.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from utils.data_processing import top_terms

SHORT = "Short-form (<=3 min)"
NICHE_KEYWORDS = {
    "gaming": ["game", "gaming", "gameplay", "minecraft", "fortnite", "roblox", "valorant", "stream", "walkthrough", "esports"],
    "sports": ["football", "soccer", "goal", "goals", "match", "league", "nba", "cricket", "player", "highlights", "training"],
    "tech": ["tech", "phone", "iphone", "android", "review", "software", "coding", "python", "ai", "laptop", "gadget"],
    "education": ["learn", "tutorial", "guide", "lesson", "explained", "how", "course", "study", "beginner"],
    "finance": ["money", "invest", "stock", "crypto", "finance", "business", "income", "trading", "budget"],
    "food": ["recipe", "cooking", "food", "kitchen", "bake", "coffee", "espresso", "meal"],
    "travel": ["travel", "trip", "vlog", "tour", "hotel", "flight", "visit"],
    "fitness": ["workout", "fitness", "gym", "health", "yoga", "training", "diet"],
    "music": ["music", "song", "cover", "guitar", "piano", "beat", "remix"],
    "beauty": ["makeup", "skincare", "beauty", "fashion", "hair", "style"],
    "comedy": ["funny", "comedy", "prank", "meme", "skit", "reaction"],
}


def detect_niche(df: pd.DataFrame) -> str:
    text = " ".join(df["title"].str.lower().tolist()) + " " + " ".join(t.lower() for tags in df["tags"] for t in (tags or []) if isinstance(t, str))
    words = Counter(text.replace(",", " ").split())
    scores = {n: sum(words[k] for k in kws) for n, kws in NICHE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "general"


def build_context(results: dict, trending_terms: list[str] | None = None) -> dict:
    """Everything the generators need, computed once from stored analysis results (no API calls)."""
    df = results["videos"]
    ctx: dict = {"channel_name": results["channel"].get("title", ""), "subscribers": results["channel"].get("subscribers"),
                 "n_videos": len(df), "trending_terms": trending_terms or []}
    if df.empty:
        return {**ctx, "available": False}
    mature = df[df["age_days"] >= 14]
    base = mature if len(mature) >= 5 else df
    ctx["available"] = True
    ctx["niche"] = detect_niche(df)
    ctx["keywords"] = [k for k, _ in top_terms(df["title"], 12)]
    tag_counts = Counter(t.lower().strip() for tags in df["tags"] for t in (tags or []) if isinstance(t, str) and t.strip())
    ctx["top_tags"] = [t for t, _ in tag_counts.most_common(15)]
    # audience-behaviour proxies (public)
    fmt = base.groupby("format").agg(videos=("video_id", "count"), median_views=("views", "median"), median_vpd=("views_per_day", "median"),
                                    engagement=("engagement_rate", "median")).reset_index()
    ctx["format_table"] = fmt
    good = fmt[fmt["videos"] >= 3]
    ctx["preferred_format"] = good.sort_values("median_vpd", ascending=False).iloc[0]["format"] if len(good) >= 2 else (
        fmt.sort_values("videos", ascending=False).iloc[0]["format"] if len(fmt) else None)
    dur = base.groupby("duration_bucket").agg(videos=("video_id", "count"), median_vpd=("views_per_day", "median")).reset_index()
    dur = dur[dur["videos"] >= 3]
    ctx["best_duration_bucket"] = dur.sort_values("median_vpd", ascending=False).iloc[0]["duration_bucket"] if len(dur) else None
    ctx["median_engagement"] = float(base["engagement_rate"].median()) if base["engagement_rate"].notna().any() else None
    ctx["comment_rate"] = float(base["comment_rate"].median()) if base["comment_rate"].notna().any() else None
    pat = results.get("title_patterns")
    ctx["pattern_perf"] = {}
    if pat is not None and len(pat):
        p = pat[pat["feature"] == "Title pattern"]
        ctx["pattern_perf"] = {r.group: (float(r.median_relative), int(r.videos)) for r in p.itertuples()}
    best_pat = [(g, v) for g, v in ctx["pattern_perf"].items() if v[1] >= 3]
    ctx["best_pattern"] = max(best_pat, key=lambda kv: kv[1][0])[0] if best_pat else None
    bucket = df["title_length_bucket"].value_counts() if "title_length_bucket" in df else pd.Series(dtype=int)
    lb = df[df["title_length_bucket"].isin(bucket[bucket >= 3].index)].groupby("title_length_bucket")["relative_to_median"].median() if len(bucket) else pd.Series(dtype=float)
    ctx["best_length_bucket"] = lb.idxmax() if len(lb) else None
    ctx["topics"] = results.get("topics", pd.DataFrame())
    ctx["own_titles"] = df["title"].tolist()
    ctx["short_share"] = float((df["format"] == SHORT).mean())
    return ctx


def audience_summary(ctx: dict) -> list[str]:
    """Plain-language statements of observed public patterns."""
    if not ctx.get("available"):
        return []
    out = []
    if ctx.get("preferred_format"):
        out.append(f"Your {ctx['preferred_format'].split(' (')[0].lower()} videos gain views fastest (median views/day among videos 14+ days old).")
    if ctx.get("best_duration_bucket"):
        out.append(f"The {ctx['best_duration_bucket']} length bucket has your highest median views/day.")
    if ctx.get("best_pattern"):
        rel, n = ctx["pattern_perf"][ctx["best_pattern"]]
        out.append(f"'{ctx['best_pattern']}' titles are your best-performing pattern ({n} videos at {rel:.2f}x your median).")
    if ctx.get("median_engagement") is not None:
        out.append(f"Typical public engagement (likes + comments / views): {ctx['median_engagement']:.2%}.")
    return out


# ----------------------------------------------------------------------------- orchestrator
def run_studio(results: dict, inputs: dict, probe: dict | None = None, provider=None, trending_terms: list[str] | None = None,
               reference_titles: list[str] | None = None, now_local=None) -> dict:
    """Run every Video Studio feature for one planned video. `inputs`: title, topic, summary, description, format ('Short'|'Long-form'|'Auto'),
    tz, music, footage, commentary, lyrics_or_scripts, ai_generated, templated, reupload, goal, music_credit, weekday (optional)."""
    from datetime import datetime

    from src import copy_generator as cg
    from src import copyright_risk as cr
    from src import editing_advice as ea
    from src import shorts_predictor as sp
    from src import upload_planner as up

    ctx = build_context(results, trending_terms)
    topic = (inputs.get("topic") or inputs.get("title") or "").strip()
    fmt = inputs.get("format", "Auto")
    dur = (probe or {}).get("duration_seconds")
    if fmt == "Auto":
        fmt = "Short" if (dur is not None and dur <= 180 and (probe or {}).get("orientation") == "vertical") or (dur is not None and dur <= 60) else "Long-form"
    out: dict = {"inputs": {k: v for k, v in inputs.items()}, "format": fmt, "context": {k: v for k, v in ctx.items() if k not in ("topics", "own_titles", "format_table")},
                 "audience_summary": audience_summary(ctx), "probe": probe}
    # 1) editing + music
    music_status = inputs.get("music_status") or {"unsure": "not decided"}.get(inputs.get("music", "unsure"), inputs.get("music", "unsure"))
    out["editing_advice"] = ea.advise(probe, ctx, fmt, music_status)
    out["music"] = ea.music_recommendation(ctx, fmt == "Short", (probe or {}).get("tempo"))
    # 2) title / caption / hashtags
    extra, ai_note = [], ""
    if provider is not None and getattr(provider, "is_configured", False) and topic:
        try:
            extra = cg.ai_titles(provider, topic, inputs.get("summary", ""), ctx, fmt)
        except Exception as exc:  # noqa: BLE001 - AI is optional
            ai_note = f"AI titles unavailable ({exc}); templates used."
    titles = cg.rank_titles(topic, ctx, inputs.get("summary", ""), fmt, extra) if topic else pd.DataFrame()
    if inputs.get("title", "").strip():
        mine = pd.DataFrame([cg.score_title(inputs["title"].strip(), ctx, set())]).assign(source="Your draft")
        titles = pd.concat([mine, titles], ignore_index=True) if len(titles) else mine
    out["titles"], out["ai_note"] = titles, ai_note
    tags = cg.hashtags(topic or inputs.get("title", ""), ctx, fmt) if (topic or inputs.get("title")) else {"hashtags": [], "above_title": [], "hidden_tags": "", "note": ""}
    out["hashtags"] = tags
    best_title = titles.sort_values("fit_score", ascending=False).iloc[0]["title"] if len(titles) else inputs.get("title", "")
    out["caption"] = cg.caption(topic, best_title, inputs.get("summary", ""), fmt, tags["hashtags"], inputs.get("goal", "subscribers"), inputs.get("music_credit", ""))
    # 3) Shorts predictor
    if fmt == "Short":
        out["shorts"] = sp.predict_short(results, best_title or topic, float(dur or 30), inputs.get("weekday"), None)
    else:
        out["shorts"] = {"available": False, "reason": "Planned as long-form; the Shorts predictor only runs for Shorts."}
    # 4) upload planner
    tz = inputs.get("tz", "UTC")
    slots = up.slot_table(results["videos"], tz, results.get("competitor_videos"))
    rec = up.recommend(slots)
    rel_lvl, rel_txt = up.reliability(results["videos"])
    gaps = results.get("gaps")
    ideas = [(best_title or topic, "Short" if fmt == "Short" else "Long-form")]
    if gaps is not None and len(gaps):
        ideas += [(g.suggested_title, g.format) for g in gaps.head(5).itertuples()]
    try:
        cad = float(results["roadmap"]["kpis"].set_index("key").loc["uploads_per_30_days", "target_30"])
        if cad != cad:
            raise KeyError
    except (KeyError, TypeError):
        cad = 8.0
    now_local = now_local or datetime.now(up.tz_of(tz))
    out["planner"] = {"slots": slots, "best": rec["best"], "avoid": rec["avoid"], "reliability": rel_lvl, "reliability_note": rel_txt,
                      "plan": up.plan_uploads(slots, cad, ideas, now_local), "tz": tz, "uploads_per_30_days_goal": cad}
    # 5) copyright / reuse risk
    rep = results.get("repeated_structures")
    share = float(rep["videos"].max() / len(results["videos"])) if rep is not None and len(rep) and len(results["videos"]) else None
    out["copyright"] = cr.screen(inputs.get("title", "") or best_title, inputs.get("description", "") or inputs.get("summary", ""), None,
                                 {"music": inputs.get("music", "unsure"), "footage": inputs.get("footage", "unsure"), "commentary": inputs.get("commentary", False),
                                  "lyrics_or_scripts": inputs.get("lyrics_or_scripts", False), "ai_generated": inputs.get("ai_generated", False),
                                  "templated": inputs.get("templated", False), "reupload": inputs.get("reupload", False)},
                                 ctx.get("own_titles", []), reference_titles, share)
    return out
