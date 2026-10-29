"""What a planned video might deliver: views, new subscribers, engagement and watch time - as RANGES with stated assumptions.

Public YouTube data does not include per-video subscribers gained, watch time or retention. So:
* views come from your own history in the same format (see shorts_predictor.predict_views, with a back-test);
* new subscribers = views x your channel's lifetime subscribers-per-view ratio (range from similar channels' ratios) - a crude average;
* likes/comments use your own like/comment rates;
* watch time uses an AVERAGE-PERCENT-WATCHED ASSUMPTION that you can edit - it is not measured.
Nothing here is a forecast or a promise. After publishing, compare with the real numbers in YouTube Studio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PCT_WATCHED = {"short_60": 0.75, "short_180": 0.55, "long_10": 0.40, "long_20": 0.32, "long_more": 0.28}
SHORT = "Short-form (<=3 min)"


def default_pct_watched(fmt: str, duration_seconds: float) -> float:
    if fmt == "Short":
        return DEFAULT_PCT_WATCHED["short_60"] if duration_seconds <= 60 else DEFAULT_PCT_WATCHED["short_180"]
    return DEFAULT_PCT_WATCHED["long_10"] if duration_seconds <= 600 else DEFAULT_PCT_WATCHED["long_20"] if duration_seconds <= 1200 else DEFAULT_PCT_WATCHED["long_more"]


def _rates(results: dict, fmt: str) -> dict:
    df = results["videos"]
    label = SHORT if fmt == "Short" else "Long-form (>3 min)"
    mature = df[(df["format"] == label) & (df["age_days"] >= (14 if fmt == "Short" else 28))]
    if len(mature) < 5:
        mature = df[df["age_days"] >= 14]
    out = {}
    for k, col in (("like", "like_rate"), ("comment", "comment_rate")):
        s = mature[col].dropna()
        out[k] = (float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75)), len(s)) if len(s) >= 3 else None
    return out


def subs_per_view(results: dict) -> tuple[float, float, float, str]:
    """(low, mid, high, note) subscribers gained per view, from lifetime channel totals."""
    ch = results["channel"]
    subs, views = ch.get("subscribers"), ch.get("total_views")
    own = subs / views if subs and views else None
    peers = []
    for c in (results.get("competitor_channels") or {}).values():
        if c.get("subscribers") and c.get("total_views"):
            peers.append(c["subscribers"] / c["total_views"])
    if own is None and len(peers) >= 3:
        own = float(np.median(peers))
    if own is None:
        return (np.nan, np.nan, np.nan, "Subscriber count or total views are not public, so subscriber gain cannot be estimated.")
    if len(peers) >= 3:
        lo, hi = float(np.percentile(peers, 25)), float(np.percentile(peers, 75))
        return (min(lo, own), own, max(hi, own), f"Your lifetime subscribers-per-view ratio ({own:.2%}); range from {len(peers)} similar channels' ratios.")
    return (own * 0.5, own, own * 1.5, f"Your lifetime subscribers-per-view ratio ({own:.2%}) with a +/-50% range (too few similar channels).")


def forecast(results: dict, views: dict, fmt: str, duration_seconds: float, pct_watched: float | None = None) -> dict:
    """Combine a views estimate (from predict_views) with conversion/engagement/watch-time assumptions."""
    if not views.get("available"):
        return {"available": False, "reason": views.get("reason", "Views estimate unavailable.")}
    v = {"low": views["estimate_low"], "mid": views["estimate_mid"], "high": views["estimate_high"]}
    r_lo, r_mid, r_hi, sub_note = subs_per_view(results)
    pct = pct_watched if pct_watched is not None else default_pct_watched(fmt, duration_seconds)
    rates = _rates(results, fmt)
    rows = [{"metric": "Views at maturity", "low": v["low"], "mid": v["mid"], "high": v["high"], "basis": f"{views['method']} (confidence {views['confidence']})"}]
    if np.isfinite(r_mid):
        rows.append({"metric": "New subscribers", "low": v["low"] * r_lo, "mid": v["mid"] * r_mid, "high": v["high"] * r_hi, "basis": sub_note + " Per-video subscriber data is private; treat as rough."})
    for key, label in (("like", "Likes"), ("comment", "Comments")):
        r = rates.get(key)
        if r:
            rows.append({"metric": label, "low": v["low"] * r[0], "mid": v["mid"] * r[1], "high": v["high"] * r[2],
                         "basis": f"Your median {label.lower()} rate {r[1]:.2%} (25th-75th pct {r[0]:.2%}-{r[2]:.2%}, {r[3]} videos)"})
    hours = lambda x: x * duration_seconds * pct / 3600  # noqa: E731
    rows.append({"metric": "Watch hours (assumption-based)", "low": hours(v["low"]), "mid": hours(v["mid"]), "high": hours(v["high"]),
                 "basis": f"Assumes viewers watch {pct:.0%} of a {duration_seconds:.0f}s video on average - NOT measured; edit the assumption"})
    tbl = pd.DataFrame(rows)
    subs_now = results["channel"].get("subscribers")
    mid_subs = float(tbl.loc[tbl["metric"] == "New subscribers", "mid"].iloc[0]) if (tbl["metric"] == "New subscribers").any() else None
    ctx = []
    if mid_subs is not None and subs_now:
        ctx.append(f"Typical case adds about {mid_subs / subs_now:.2%} to your current subscriber count.")
    med = results["videos"]["views"].median()
    if med:
        ctx.append(f"Typical views are {v['mid'] / med:.2f}x your overall median video.")
    return {"available": True, "table": tbl, "pct_watched": pct, "assumptions": {"pct_watched": pct, "subs_per_view": (r_lo, r_mid, r_hi)}, "context": ctx,
            "notes": ["Ranges, not forecasts: individual videos often fall outside them.", "Watch time and subscriber gain are not public per video, so those rows rest on stated assumptions.",
                      "After publishing, compare with YouTube Studio's real numbers and adjust."]}
