"""Growth roadmap: benchmark KPIs vs auto-discovered peers, a transparent health score, planning targets for
day 30/60/90, a day-by-day 30-day action plan, and simple scenario arithmetic.

IMPORTANT: targets are *planning goals* derived from peer benchmarks, not predictions. Nothing here promises views,
subscribers or ranking. Outcome KPIs (views etc.) depend on factors outside your control.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src import competitor_analysis as comp_a
from src.content_strategy import FALLBACK_TOPICS, _post_offsets

KPI_SPECS = [
    {"key": "uploads_per_30_days", "label": "Uploads per 30 days", "higher": True, "weight": 0.20, "controllable": True, "fmt": "{:.1f}"},
    {"key": "gap_cv", "label": "Upload steadiness (gap variation; lower = steadier)", "higher": False, "weight": 0.10, "controllable": True, "fmt": "{:.2f}"},
    {"key": "median_views_per_day", "label": "Median views per day per video", "higher": True, "weight": 0.20, "controllable": False, "fmt": "{:,.1f}"},
    {"key": "median_views_per_1k_subs", "label": "Median views per 1,000 subscribers", "higher": True, "weight": 0.20, "controllable": False, "fmt": "{:,.1f}"},
    {"key": "median_engagement_rate", "label": "Median engagement rate (likes+comments / views)", "higher": True, "weight": 0.15, "controllable": False, "fmt": "{:.2%}"},
    {"key": "breakout_rate", "label": "Breakout rate (videos at 1.5x+ your median)", "higher": True, "weight": 0.15, "controllable": False, "fmt": "{:.0%}"},
]
FALLBACK_BENCHMARK = {"uploads_per_30_days": 4.0, "gap_cv": 0.6}  # planning defaults used only when no peers are available
FRACTIONS_CONTROLLABLE = (0.5, 1.0, 1.0)  # share of the gap to close by day 30 / 60 / 90
FRACTIONS_OUTCOME = (0.20, 0.35, 0.50)
OUTCOME_CAP = 1.0  # outcome targets never exceed +100% of the current value in 90 days


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def build_kpi_table(own_row: dict, peers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in KPI_SPECS:
        k = spec["key"]
        you = own_row.get(k)
        peer_vals = pd.to_numeric(peers[k], errors="coerce").dropna() if len(peers) and k in peers else pd.Series(dtype=float)
        if len(peer_vals) >= 3:
            bench, top_q, src = float(peer_vals.median()), float(peer_vals.quantile(0.75 if spec["higher"] else 0.25)), f"Peer median (n={len(peer_vals)})"
        elif k in FALLBACK_BENCHMARK:
            bench = top_q = FALLBACK_BENCHMARK[k]
            src = "Planning default (too few peers)"
        else:
            bench = top_q = np.nan
            src = "No benchmark available"
        row = {"key": k, "metric": spec["label"], "you": you if _ok(you) else np.nan, "benchmark": bench, "top_quartile": top_q,
               "benchmark_source": src, "weight": spec["weight"], "controllable": spec["controllable"], "higher_is_better": spec["higher"]}
        att = np.nan
        if _ok(you) and _ok(bench):
            if spec["higher"]:
                att = min(1.0, you / bench) if bench > 0 else 1.0
            else:
                att = min(1.0, bench / you) if you > 0 else 1.0
        row["attainment"] = att
        if _ok(att):
            goal = bench if att < 1 else (top_q if _ok(top_q) else you)
            if att >= 1 and spec["higher"] and goal <= you:
                goal = you * 1.10
            if att >= 1 and not spec["higher"] and goal >= you:
                goal = you * 0.90
            fr = FRACTIONS_CONTROLLABLE if spec["controllable"] else FRACTIONS_OUTCOME
            targets = []
            for f in fr:
                t = you + f * (goal - you)
                if not spec["controllable"] and spec["higher"]:
                    t = min(t, you * (1 + OUTCOME_CAP))
                targets.append(float(t))
            row.update(goal=float(goal), target_30=targets[0], target_60=targets[1], target_90=targets[2])
            row["status"] = "Ahead of peers" if att >= 1 else f"{(1 - att):.0%} below peer benchmark"
        else:
            row.update(goal=np.nan, target_30=np.nan, target_60=np.nan, target_90=np.nan, status="Not enough data")
        row["priority"] = round(spec["weight"] * (1 - att), 4) if _ok(att) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def health_score(kpis: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    """0-100 = weighted attainment vs benchmark (100 means at or above the benchmark on every KPI)."""
    d = kpis[kpis["attainment"].notna()].copy()
    if d.empty:
        return None, pd.DataFrame()
    total_w = d["weight"].sum()
    d["points"] = d["weight"] / total_w * d["attainment"] * 100
    d["max_points"] = d["weight"] / total_w * 100
    return round(float(d["points"].sum()), 1), d[["metric", "weight", "attainment", "points", "max_points"]].reset_index(drop=True)


def momentum(df: pd.DataFrame) -> dict:
    """Newer half vs older half of the sample. Newer videos have had less time to gain views, so this is conservative."""
    if len(df) < 6:
        return {"available": False}
    cut = df["published_at"].median()
    new, old = df[df["published_at"] >= cut], df[df["published_at"] < cut]
    a, b = float(new["views_per_day"].median()), float(old["views_per_day"].median())
    return {"available": True, "recent_views_per_day": a, "older_views_per_day": b,
            "change_pct": (a / b - 1) if b > 0 else np.nan,
            "note": "Compares median views/day of the newer vs older half of your sample; very new videos are still accumulating views."}


def _evidence(own_df, topics, title_patterns, patterns, format_cmp) -> dict:
    ev: dict = {"topic": None, "pattern": None, "weekday": None, "format": None, "breakouts": []}
    if len(topics):
        t = topics[(topics["videos"] >= 2) & (~topics["topic"].str.startswith("Other"))]
        if len(t):
            r = t.sort_values("median_relative", ascending=False).iloc[0]
            ev["topic"] = (r["topic"], float(r["median_relative"]), int(r["videos"]))
    if len(title_patterns):
        p = title_patterns[(title_patterns["feature"] == "Title pattern") & (title_patterns["videos"] >= 3)]
        if len(p):
            r = p.sort_values("median_relative", ascending=False).iloc[0]
            ev["pattern"] = (r["group"], float(r["median_relative"]), int(r["videos"]))
    wd = patterns.get("weekday") if isinstance(patterns, dict) else None
    if wd is not None and len(wd) and (wd["videos"] >= 2).any():
        r = wd[wd["videos"] >= 2].sort_values("median_views", ascending=False).iloc[0]
        ev["weekday"] = (r["weekday"], float(r["median_views"]), int(r["videos"]))
    if len(format_cmp) and (format_cmp["videos"] >= 3).sum() >= 2:
        r = format_cmp[format_cmp["videos"] >= 3].sort_values("median_views_per_day", ascending=False).iloc[0]
        ev["format"] = (r["format"], float(r["median_views_per_day"]), int(r["videos"]))
    if len(own_df):
        ev["breakouts"] = own_df.sort_values("relative_to_median", ascending=False).head(3)["title"].tolist()
    return ev


def _fmt(spec_key: str, v: float) -> str:
    spec = next(s for s in KPI_SPECS if s["key"] == spec_key)
    return spec["fmt"].format(v) if _ok(v) else "n/a"


def build_actions(kpis: pd.DataFrame, ev: dict) -> pd.DataFrame:
    acts = []
    for r in kpis.sort_values("priority", ascending=False).itertuples():
        if not _ok(r.attainment) or r.attainment >= 1:
            continue
        k, you = r.key, r.you
        why = (f"You: {_fmt(k, you)} vs {r.benchmark_source.lower()}: {_fmt(k, r.benchmark)}. "
               f"Targets (planning goals): day 30 {_fmt(k, r.target_30)}, day 60 {_fmt(k, r.target_60)}, day 90 {_fmt(k, r.target_90)}.")
        label = "Peer benchmark" if r.benchmark_source.startswith("Peer") else "Planning default"
        if k == "uploads_per_30_days":
            head = f"Raise publishing cadence to about {r.target_30:.1f} uploads per 30 days, then {r.target_90:.1f}"
            how = ["Batch-produce: script and film 2-3 videos in one session", f"Fix publishing days (best observed weekday: {ev['weekday'][0]}, small sample)" if ev["weekday"] else "Fix two publishing days per week and keep them",
                   "Keep a backlog of at least 5 ideas (see Content Gaps)"]
        elif k == "gap_cv":
            head = "Publish on a steadier schedule"
            how = ["Choose fixed publishing days and times", "Schedule uploads in advance (YouTube Studio) so busy weeks don't create long gaps", "Review gaps weekly: no gap longer than your median"]
        elif k in ("median_views_per_day", "median_views_per_1k_subs"):
            head = "Lift per-video pace through topic and packaging"
            how = []
            if ev["topic"]:
                how.append(f"Make more of your '{ev['topic'][0]}' topic ({ev['topic'][2]} videos at {ev['topic'][1]:.2f}x your median)")
            if ev["pattern"]:
                how.append(f"Test more '{ev['pattern'][0]}' titles ({ev['pattern'][2]} videos at {ev['pattern'][1]:.2f}x your median; observational)")
            how += ["Rewrite titles/thumbnails of 3 older videos that still get traffic", "Use top Content Gap ideas as test videos"]
        elif k == "median_engagement_rate":
            head = "Increase viewer interaction (hypothesis to test)"
            how = ["Ask one specific question in the first 30 seconds and near the end", "Pin a comment that repeats the question; reply to early comments",
                   "Compare engagement on the next 5 videos with your baseline"]
            label = label + " / creative hypothesis"
        else:
            head = "Make breakout videos more repeatable"
            how = ["Study your top videos: " + "; ".join(ev["breakouts"][:3]) if ev["breakouts"] else "Study your top videos",
                   "List what they share (topic, hook, format, title structure)", "Turn the shared pattern into a 3-part series"]
        acts.append({"priority_rank": len(acts) + 1, "kpi": r.metric, "action": head, "why": why, "how": " | ".join(how), "evidence_type": label})
    if not acts:
        acts.append({"priority_rank": 1, "kpi": "All KPIs", "action": "You are at or above the benchmark on every KPI: protect consistency and test stretch goals",
                     "why": "Targets are set to the top quartile of peers.", "how": "Keep cadence | Test one new format per month", "evidence_type": "Peer benchmark"})
    return pd.DataFrame(acts)


TASK_CYCLE = [
    ("Research", "Collect 3 angles from top competitor/peer videos on your next topic (note what they miss)", "Views per day", 45, "Angle notes"),
    ("Script", "Write the script outline and 3 alternative opening hooks", "Breakout rate", 60, "Script + 3 hooks"),
    ("Packaging", "Write 5 title options (clear subject first, no overpromising) and sketch the thumbnail", "Views per 1,000 subscribers", 40, "Title shortlist + thumbnail sketch"),
    ("Engage", "Reply to recent comments and pin a question on your latest video", "Engagement rate", 30, "Pinned comment + replies"),
    ("Repurpose", "Cut a Short or clip from your best recent long video", "Uploads per 30 days", 60, "One Short/clip"),
    ("Produce", "Film or record the next video (batch if possible)", "Uploads per 30 days", 120, "Raw footage"),
]


def build_daily_plan(kpis: pd.DataFrame, own_df: pd.DataFrame, gaps: pd.DataFrame, ev: dict, start: date, days: int = 30) -> pd.DataFrame:
    kp = kpis.set_index("key")
    cadence = kp.loc["uploads_per_30_days", "target_30"] if _ok(kp.loc["uploads_per_30_days", "target_30"]) else kp.loc["uploads_per_30_days", "you"]
    cadence = cadence if _ok(cadence) else 4.0
    per_week = float(np.clip(cadence / 30 * 7, 1, 7))
    upload_days = set(_post_offsets(days, per_week))
    ideas = [(g.suggested_title, g.format) for g in gaps.itertuples()] if len(gaps) else []
    ideas += [(t, "Long-form (>3 min)") for t in FALLBACK_TOPICS]
    top3 = "; ".join(own_df.sort_values("views", ascending=False).head(3)["title"]) if len(own_df) else "your top videos"
    bottom = own_df[own_df["performance_flag"] == "Below typical"].head(3)["title"].tolist() if len(own_df) and "performance_flag" in own_df else []
    rows, idea_i, cyc_i, uploads = [], 0, 0, 0
    for d in range(days):
        day, dt = d + 1, start + timedelta(days=d)
        phase = "Foundation (days 1-7)" if day <= 7 else "Build (days 8-21)" if day <= 21 else "Optimise (days 22-30)"
        if day == 1:
            focus, task, kpi, mins, deliv = "Baseline", "Record today's baseline: subscribers, median views, views/day, uploads in the last 30 days (see KPI table)", "All KPIs", 30, "Baseline sheet"
        elif day == 2:
            focus, task, kpi, mins, deliv = "Audit", f"Review your top videos ({top3}) and list what they share: topic, hook, format, title structure", "Breakout rate", 60, "Pattern list"
        elif day == 3:
            focus, task, kpi, mins, deliv = "Plan", "Pick 3 topic pillars from your strongest topics and the top Content Gap ideas; fix your publishing days", "Upload steadiness", 45, "Topic pillars + schedule"
        elif day == 4 and bottom:
            focus, task, kpi, mins, deliv = "Quick win", f"Rewrite titles/descriptions for older below-typical videos (e.g. {'; '.join(bottom[:2])}); keep the video's real subject", "Views per day", 45, "Updated titles"
        elif day % 7 == 0:
            ups = int(round(cadence * day / 30))
            focus, task, kpi, mins, deliv = "Weekly checkpoint", (f"Compare with targets: about {ups} upload(s) should be live by now (day-30 goal {_fmt('uploads_per_30_days', cadence)} per 30 days); "
                                                                  "note views/day of the last 7 days and adjust next week's topics"), "All KPIs", 30, "Checkpoint notes"
        elif d in upload_days:
            title, fmt = ideas[idea_i % len(ideas)]
            idea_i += 1
            uploads += 1
            focus, task, kpi, mins, deliv = "Publish", f"Publish video #{uploads}: \"{title}\" ({fmt}). Pre-publish check: subject clear in the first words of the title, no overpromising, one clear question or CTA", "Uploads per 30 days", 60, "Published video"
        else:
            focus, task, kpi, mins, deliv = (lambda c: (c[0], c[1], c[2], c[3], c[4]))(TASK_CYCLE[cyc_i % len(TASK_CYCLE)])
            cyc_i += 1
        rows.append({"day": day, "date": dt.isoformat(), "phase": phase, "focus": focus, "task": task, "kpi": kpi, "est_minutes": mins, "deliverable": deliv, "done": False})
    return pd.DataFrame(rows)


def build_phases(kpis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col, focus in (("Day 30", "target_30", "Foundation: baseline, cadence and schedule in place; first packaging tests"),
                              ("Day 60", "target_60", "Build: repeat what worked; steady schedule; improve engagement habits"),
                              ("Day 90", "target_90", "Optimise: double down on proven topics/formats; reassess against peers")):
        for r in kpis.itertuples():
            if _ok(getattr(r, col)):
                rows.append({"milestone": label, "focus": focus, "metric": r.metric, "current": _fmt(r.key, r.you), "target": _fmt(r.key, getattr(r, col))})
    return pd.DataFrame(rows)


def build_scenarios(kpis: pd.DataFrame, own_median_views: float, uplift: float = 0.10) -> pd.DataFrame:
    """Planning arithmetic: videos published in 90 days x assumed median views per video. NOT a forecast."""
    kp = kpis.set_index("key")
    cur = kp.loc["uploads_per_30_days", "you"]
    if not _ok(cur) or not _ok(own_median_views):
        return pd.DataFrame()
    t = [kp.loc["uploads_per_30_days", c] for c in ("target_30", "target_60", "target_90")]
    target_videos = float(sum(x if _ok(x) else cur for x in t))
    base_videos = float(cur * 3)
    rows = [
        {"scenario": "Maintain current pace", "videos_in_90_days": base_videos, "assumed_median_views": own_median_views},
        {"scenario": "Reach cadence targets", "videos_in_90_days": target_videos, "assumed_median_views": own_median_views},
        {"scenario": f"Cadence targets + packaging uplift ({uplift:+.0%} assumed)", "videos_in_90_days": target_videos, "assumed_median_views": own_median_views * (1 + uplift)},
    ]
    df = pd.DataFrame(rows)
    df["views_from_new_videos"] = df["videos_in_90_days"] * df["assumed_median_views"]
    df["note"] = "Planning arithmetic (videos x median views); not a forecast. New videos need time to accumulate views; the uplift is an assumption you can change."
    return df


def build_roadmap(own_df: pd.DataFrame, channel: dict, comparison: pd.DataFrame, topics: pd.DataFrame, title_patterns: pd.DataFrame,
                  patterns: dict, gaps: pd.DataFrame, format_cmp: pd.DataFrame, start: date | None = None, uplift: float = 0.10) -> dict:
    if own_df.empty:
        return {"available": False}
    start = start or datetime.now(timezone.utc).date()
    own_row = comp_a.summarize_channel_row("You", channel, own_df)
    peers = comparison[comparison["role"] == "Competitor"] if len(comparison) else pd.DataFrame()
    kpis = build_kpi_table(own_row, peers)
    score, comps = health_score(kpis)
    ev = _evidence(own_df, topics, title_patterns, patterns, format_cmp)
    return {
        "available": True,
        "benchmark_source": f"{len(peers)} auto-discovered peer channels" if len(peers) else "No peers available - planning defaults only",
        "kpis": kpis, "health_score": score, "health_components": comps, "momentum": momentum(own_df),
        "actions": build_actions(kpis, ev), "daily_plan": build_daily_plan(kpis, own_df, gaps, ev, start),
        "phases": build_phases(kpis), "scenarios": build_scenarios(kpis, float(own_df["views"].median()), uplift),
        "evidence": {k: v for k, v in ev.items()},
        "notes": ["Targets are planning goals derived from peer benchmarks, not predictions or promises.",
                  "Peers are auto-discovered by topic match and differ from you in age, niche mix and audience; treat gaps as hypotheses to test.",
                  "Outcome KPIs (views, engagement) also depend on factors you cannot control."],
    }
