"""Orchestrates the full analysis and returns one results dict used by the UI and every exporter."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src import channel_analysis as ca
from src import competitor_analysis as comp_a
from src import competitor_discovery as disc
from src import content_gap_analysis as gaps_a
from src import audience_profile as aprof
from src import monetization as mon
from src import roadmap as rm
from src import title_analysis as ta
from src import topic_analysis as topa
from src import video_analysis as va
from src.youtube_api import YouTubeAPIError
from utils.data_processing import videos_to_df
from utils.validation import ParsedChannel, parse_channel_input, parse_many

BASE_LIMITATIONS = [
    "All figures come from public YouTube Data API v3 fields. Watch time, impressions, click-through rate, audience retention, traffic sources and revenue are NOT available unless a channel owner authorises YouTube Analytics access (not used here). The Income tab therefore shows clearly-labelled ESTIMATES from public data and editable assumptions, never actual revenue.",
    "Lifetime channel statistics (whole channel) are reported separately from statistics of the selected video sample/date range.",
    "Views are cumulative, so older videos have had longer to accumulate; a low-view video is not automatically a failure.",
    "Topic groups come from titles, tags and the start of descriptions (no transcripts); they are approximate.",
    "A content gap is not proof of search demand; no search-volume data was used.",
    "AI-generated titles and plans (if enabled) are recommendations, not guaranteed improvements. Nothing here predicts views, subscribers, virality or ranking.",
]


def _prepare(bundle: dict, now: datetime | None) -> tuple[dict, pd.DataFrame]:
    df = videos_to_df(bundle["videos"])
    df = va.enrich_videos(df, now)
    return bundle["channel"], df


def run_analysis(client: Any, channel_input: str, competitor_inputs: list[str] | None = None, sample_size: int = 50,
                 date_from: datetime | None = None, date_to: datetime | None = None,
                 topic_mapping: dict[str, list[str]] | None = None, now: datetime | None = None,
                 auto_competitors: int = 0, competitor_sample_size: int | None = None,
                 progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Run everything. Raises ValueError (bad input) or YouTubeAPIError (API problem) for the main channel;
    competitor failures are recorded as warnings so the rest of the analysis still completes."""
    step = progress or (lambda frac, msg: None)
    now = now or datetime.now(timezone.utc)
    own_ref = parse_channel_input(channel_input)
    comp_refs, input_errors = parse_many(competitor_inputs or [], limit=5)
    warnings: list[str] = list(input_errors)
    comp_sample = competitor_sample_size or min(sample_size, 40)

    step(0.05, "Fetching your channel and videos...")
    bundle = client.fetch_channel_bundle(own_ref, sample_size, date_from, date_to)
    channel, own_df = _prepare(bundle, now)
    notes = list(bundle.get("notes", []))

    step(0.35, "Analysing videos, titles and topics...")
    own_df = ta.analyze_titles(own_df)
    own_df = topa.assign_topics(own_df, topic_mapping)
    topic_tbl = topa.topic_summary(own_df)
    summary = ca.summarize_channel(channel, own_df)

    discovered = pd.DataFrame()
    if auto_competitors > 0:
        step(0.38, "Finding similar channels automatically...")
        try:
            refs, discovered, d_notes = disc.discover_competitors(client, channel, own_df, topic_tbl, n=auto_competitors, now=now)
            notes += d_notes
            comp_refs = comp_refs + [r for r in refs if r.value not in {c.value for c in comp_refs}]
        except YouTubeAPIError as exc:
            warnings.append(f"Automatic competitor discovery failed: {exc}")

    competitors: dict[str, tuple[dict, pd.DataFrame]] = {}
    for k, ref in enumerate(comp_refs):
        step(0.4 + 0.35 * (k / max(len(comp_refs), 1)), f"Fetching competitor {k + 1} of {len(comp_refs)}...")
        try:
            b = client.fetch_channel_bundle(ref, comp_sample, date_from, date_to)
        except (YouTubeAPIError, ValueError) as exc:
            warnings.append(f"Competitor '{ref.value}' skipped: {exc}")
            continue
        ch, df = _prepare(b, now)
        if ch.get("channel_id") == channel.get("channel_id"):
            warnings.append(f"Competitor '{ref.value}' is the same channel as yours and was skipped.")
            continue
        if ch.get("channel_id") in {c[0].get("channel_id") for c in competitors.values()}:
            continue
        competitors[ch.get("title") or ref.value] = (ch, df)
        notes += [f"{ch.get('title')}: {n}" for n in b.get("notes", [])]

    step(0.8, "Comparing channels and finding content gaps...")
    comparison = comp_a.compare_channels((channel, own_df), competitors) if competitors else pd.DataFrame()
    gaps = gaps_a.find_content_gaps(own_df, {n: d for n, (_, d) in competitors.items()}, topic_tbl)

    is_sample = bool(getattr(client, "is_sample", False))
    results: dict[str, Any] = {
        "meta": {
            "analysis_date": now.isoformat(), "mode": "SAMPLE DATA (illustrative)" if is_sample else "live",
            "channel_input": channel_input, "competitor_inputs": [r.value for r in comp_refs], "auto_competitors_requested": auto_competitors, "sample_size_requested": sample_size,
            "date_from": date_from.isoformat() if date_from else None, "date_to": date_to.isoformat() if date_to else None,
            "topic_method": "custom keyword mapping" if topic_mapping else "automatic keyword clustering",
            "version": 1,
        },
        "channel": channel,
        "summary": summary,
        "overview_table": ca.overview_table(summary),
        "videos": own_df,
        "top_videos": va.top_videos(own_df),
        "lowest_videos": va.lowest_videos(own_df),
        "format_comparison": va.format_comparison(own_df),
        "publishing_patterns": ca.publishing_patterns(own_df),
        "title_patterns": ta.title_pattern_table(own_df),
        "similar_titles": ta.similar_titles(own_df),
        "repeated_structures": ta.repeated_structures(own_df),
        "topics": topic_tbl,
        "topic_trend": topa.recent_vs_older(own_df),
        "competitors": comparison,
        "competitor_videos": {n: d for n, (_, d) in competitors.items()},
        "competitor_channels": {n: c for n, (c, _) in competitors.items()},
        "competitor_differences": comp_a.strategy_differences(comparison) if len(comparison) else [],
        "discovered_competitors": discovered,
        "competitor_top_videos": comp_a.top_competitor_videos(competitors) if competitors else pd.DataFrame(),
        "gaps": gaps,
        "calendar": pd.DataFrame(),
        "title_suggestions": {},
        "thumbnails": pd.DataFrame(),
        "notes": notes,
        "warnings": warnings,
        "limitations": BASE_LIMITATIONS + (comp_a.LIMITATIONS if competitors else []),
        "sources": _sources(channel, competitors),
    }
    results["roadmap"] = _roadmap(results)
    results["monetization"] = _monetization(results, now=now)
    results["audience_profile"] = aprof.build_profile(results["videos"])
    step(1.0, "Done")
    return results


def _roadmap(results: dict, uplift: float = 0.10) -> dict:
    return rm.build_roadmap(results["videos"], results["channel"], results["competitors"], results["topics"], results["title_patterns"],
                            results["publishing_patterns"], results["gaps"], results["format_comparison"], uplift=uplift)


def _cadence_ratio(results: dict) -> float:
    """Day-90 upload goal / current uploads per 30 days from the roadmap (1.5 if unavailable)."""
    try:
        k = results["roadmap"]["kpis"].set_index("key").loc["uploads_per_30_days"]
        return float(k["target_90"] / k["you"]) if k["you"] and k["target_90"] == k["target_90"] else 1.5
    except (KeyError, TypeError, ZeroDivisionError):
        return 1.5


def _monetization(results: dict, assumptions: dict | None = None, now: datetime | None = None) -> dict:
    a = {"cadence_ratio": _cadence_ratio(results), **(assumptions or {})}
    return mon.build_monetization(results["channel"], results["videos"], a, now)


def refresh_monetization(results: dict, assumptions: dict) -> dict:
    """Recompute the income estimate from stored data with new assumptions (no API calls)."""
    results["monetization"] = _monetization(results, assumptions)
    results["meta"]["version"] += 1
    return results


def refresh_roadmap(results: dict, uplift: float = 0.10) -> dict:
    results["roadmap"] = _roadmap(results, uplift)
    results["monetization"] = _monetization(results, results.get("monetization", {}).get("assumptions"))
    results["meta"]["version"] += 1
    return results


def recompute_topics(results: dict, mapping: dict[str, list[str]] | None) -> dict:
    """Re-run topic grouping and gap detection from stored data (no API calls)."""
    df = topa.assign_topics(results["videos"].drop(columns=["topic"], errors="ignore"), mapping)
    results["videos"] = df
    results["topics"] = topa.topic_summary(df)
    results["topic_trend"] = topa.recent_vs_older(df)
    results["gaps"] = gaps_a.find_content_gaps(df, results["competitor_videos"], results["topics"])
    results["meta"]["topic_method"] = "custom keyword mapping" if mapping else "automatic keyword clustering"
    results["roadmap"] = _roadmap(results)
    results["monetization"] = _monetization(results, results.get("monetization", {}).get("assumptions"))
    results["meta"]["version"] += 1
    return results


def apply_manual_topics(results: dict, overrides: dict[str, str]) -> dict:
    """Apply user corrections {video_id: topic}, then refresh topic summary and content gaps (no API calls)."""
    df = results["videos"].copy()
    clean = {k: v.strip() for k, v in overrides.items() if v and v.strip()}
    df["topic"] = [clean.get(vid, t) for vid, t in zip(df["video_id"], df["topic"])]
    results["videos"] = df
    results["topics"] = topa.topic_summary(df)
    results["topic_trend"] = topa.recent_vs_older(df)
    results["gaps"] = gaps_a.find_content_gaps(df, results["competitor_videos"], results["topics"])
    results["meta"]["topic_method"] = "automatic/custom grouping with manual corrections"
    results["roadmap"] = _roadmap(results)
    results["monetization"] = _monetization(results, results.get("monetization", {}).get("assumptions"))
    results["meta"]["version"] += 1
    return results


def _sources(channel: dict, competitors: dict) -> list[dict]:
    rows = [{"type": "API", "item": "YouTube Data API v3", "detail": "https://developers.google.com/youtube/v3"},
            {"type": "Channel", "item": channel.get("title"), "detail": channel.get("url")}]
    rows += [{"type": "Competitor channel", "item": n, "detail": c.get("url")} for n, (c, _) in competitors.items()]
    return rows
