"""Your upload vs competitors: comparison metrics and the closest competitor videos (public data only)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.data_processing import term_set

SHORT = "Short-form (<=3 min)"


def _label(fmt: str) -> str:
    return SHORT if fmt == "Short" else "Long-form (>3 min)"


def _peers(results: dict, fmt: str) -> pd.DataFrame:
    frames = [d for d in (results.get("competitor_videos") or {}).values() if len(d)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["format"] == _label(fmt)]


def _verdict(x: float, p25: float, med: float, p75: float, higher_better: bool | None = None) -> str:
    if any(map(lambda v: v is None or (isinstance(v, float) and np.isnan(v)), (x, p25, med, p75))):
        return "n/a"
    if p25 <= x <= p75:
        return "Within the typical range"
    side = "Above" if x > p75 else "Below"
    if higher_better is None:
        return f"{side} the typical range"
    return f"{side} the typical range ({'good' if (x > p75) == higher_better else 'check'})"


def _q(s: pd.Series) -> tuple[float, float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return (float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))) if len(s) >= 3 else (np.nan, np.nan, np.nan)


def compare(results: dict, fmt: str, title: str, duration_seconds: float | None, views_est: dict | None, forecast: dict | None) -> pd.DataFrame:
    """Rows: metric | this upload | your median | peers 25th | peers median | peers 75th | verdict."""
    own = results["videos"]
    own = own[own["format"] == _label(fmt)]
    peers = _peers(results, fmt)
    rows = []

    def add(metric, x, own_vals, peer_vals, higher=None, note=""):
        p25, med, p75 = _q(peer_vals) if len(peer_vals) else (np.nan, np.nan, np.nan)
        om = _q(own_vals)[1] if len(own_vals) else np.nan
        rows.append({"metric": metric, "this_upload": x, "your_median": om, "peers_25th": p25, "peers_median": med, "peers_75th": p75,
                     "verdict": _verdict(x, p25, med, p75, higher) if x is not None else "n/a", "note": note})

    if duration_seconds:
        add("Length (seconds)", float(duration_seconds), own["duration_seconds"] if len(own) else [], peers["duration_seconds"] if len(peers) else [],
            note="Peers' typical length in the same format")
    if title.strip():
        add("Title length (characters)", float(len(title.strip())), own["title"].str.len() if len(own) else [], peers["title"].str.len() if len(peers) else [])
        add("Title words", float(len(title.split())), own["title"].str.split().str.len() if len(own) else [], peers["title"].str.split().str.len() if len(peers) else [])
    if views_est and views_est.get("available"):
        subs = results["channel"].get("subscribers")
        peer_rate = []
        for name, d in (results.get("competitor_videos") or {}).items():
            ch = (results.get("competitor_channels") or {}).get(name, {})
            dd = d[(d["format"] == _label(fmt)) & (d["age_days"] >= (14 if fmt == "Short" else 28))]
            if len(dd) >= 3 and ch.get("subscribers"):
                peer_rate.append(float(dd["views"].median()) / ch["subscribers"] * 1000)
        if peer_rate and subs:
            p25, med, p75 = [float(np.percentile(peer_rate, q)) * subs / 1000 for q in (25, 50, 75)]
            rows.append({"metric": "Estimated views (size-adjusted peers)", "this_upload": views_est["estimate_mid"], "your_median": views_est.get("baseline", np.nan),
                         "peers_25th": p25, "peers_median": med, "peers_75th": p75, "verdict": _verdict(views_est["estimate_mid"], p25, med, p75, True),
                         "note": "Peers' median views per 1,000 subscribers scaled to your subscriber count"})
    if forecast and forecast.get("available"):
        t = forecast["table"].set_index("metric")
        if "Likes" in t.index and "Comments" in t.index and views_est["estimate_mid"] > 0:
            er = (t.loc["Likes", "mid"] + t.loc["Comments", "mid"]) / views_est["estimate_mid"]
            add("Expected engagement rate", float(er), own["engagement_rate"] if len(own) else [], peers["engagement_rate"] if len(peers) else [], True,
                "Your own typical rates applied to this video; compared with peers' actual rates")
    pat_x = bool(re.search(r"\d", title))
    if len(peers) >= 5 and title.strip():
        top = peers.sort_values("views_per_day", ascending=False).head(max(5, len(peers) // 4))
        rows.append({"metric": "Title contains a number", "this_upload": float(pat_x), "your_median": float(own["title"].str.contains(r"\d").mean()) if len(own) else np.nan,
                     "peers_25th": np.nan, "peers_median": float(peers["title"].str.contains(r"\d").mean()), "peers_75th": float(top["title"].str.contains(r"\d").mean()),
                     "verdict": "Informational", "note": "Share of videos with a number: all peers' (median column) vs peers' fastest quarter (75th column)"})
    return pd.DataFrame(rows)


def closest_videos(results: dict, text: str, n: int = 5) -> pd.DataFrame:
    """Peer videos whose titles/tags overlap most with this video's topic - the closest direct competition."""
    want = term_set(text)
    if not want:
        return pd.DataFrame()
    rows = []
    for name, d in (results.get("competitor_videos") or {}).items():
        for r in d.itertuples():
            have = term_set(f"{r.title} . {' , '.join(r.tags) if isinstance(r.tags, list) else ''}")
            if not have:
                continue
            sim = len(want & have) / len(want | have)
            if sim > 0:
                rows.append({"similarity": round(sim, 3), "channel": name, "title": r.title, "url": r.url, "views": int(r.views), "views_per_day": float(r.views_per_day),
                             "relative_to_channel_median": float(r.relative_to_median), "age_days": float(r.age_days)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["similarity", "views_per_day"], ascending=False).head(n).reset_index(drop=True)
