"""Shorts performance estimator built from YOUR channel's own Shorts history, with an honest back-test.

What it does: baseline = median views of your mature Shorts; optional adjustments for topic, title pattern, weekday and length
(each shrunk toward 'no effect' when samples are small); an uncertainty range from how much your Shorts actually vary.
Back-test: it predicts each of your past Shorts using all the OTHER Shorts and reports how often reality fell in the range, and
whether the adjustments beat the plain baseline. If they do not, the estimate uses the baseline only.
It is an estimate of typical outcomes, not a forecast: no model can know whether a given Short will take off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.title_analysis import classify_pattern

SHORT = "Short-form (<=3 min)"
MATURE_DAYS = 14
SHRINK_K = 5.0
FACTOR_CLAMP = (0.6, 1.6)
MIN_FOR_MODEL = 8
LENGTH_BINS = [(0, 20, "up to 20s"), (20, 40, "20-40s"), (40, 60, "40-60s"), (60, 181, "1-3 min")]


def length_bucket(seconds: float) -> str:
    for lo, hi, label in LENGTH_BINS:
        if lo <= seconds < hi:
            return label
    return LENGTH_BINS[-1][2]


def _topic_of(title: str, known_topics: list[str]) -> str:
    low = title.lower()
    for t in sorted(known_topics, key=len, reverse=True):
        if t.lower() in low:
            return t
    return "Other / unclassified"


def _shrunk(group_median_rel: float, n: int) -> float:
    f = (n * group_median_rel + SHRINK_K * 1.0) / (n + SHRINK_K)
    return float(np.clip(f, *FACTOR_CLAMP))


def _features(row: dict) -> dict:
    return {"topic": row["topic"], "pattern": row["pattern"], "weekday": row["weekday"], "length": row["length"]}


def _prep(shorts: pd.DataFrame, known_topics: list[str]) -> pd.DataFrame:
    d = shorts.copy()
    d["topic"] = d["topic"] if "topic" in d else d["title"].apply(lambda t: _topic_of(t, known_topics))
    d["pattern"] = d["title"].apply(classify_pattern)
    d["weekday"] = d["published_at"].dt.dayofweek
    d["length"] = d["duration_seconds"].apply(length_bucket)
    return d


def _model(train: pd.DataFrame) -> dict:
    base = float(train["views"].median())
    rel = train["views"] / base if base > 0 else train["views"] * 0 + 1
    tab = {}
    for feat in ("topic", "pattern", "weekday", "length"):
        g = pd.DataFrame({"k": train[feat], "rel": rel}).groupby("k")["rel"].agg(["median", "count"])
        tab[feat] = {k: (_shrunk(float(r["median"]), int(r["count"])), int(r["count"])) for k, r in g.iterrows()}
    resid = (train["views"] / base).clip(lower=1e-3)
    return {"base": base, "tab": tab, "q25": float(resid.quantile(0.25)), "q75": float(resid.quantile(0.75))}


def _predict(m: dict, feats: dict, use_factors: bool) -> tuple[float, dict]:
    used = {}
    f = 1.0
    if use_factors:
        for feat, val in feats.items():
            fac, n = m["tab"].get(feat, {}).get(val, (1.0, 0))
            used[feat] = (val, fac, n)
            f *= fac
        f = float(np.clip(f, 0.4, 2.5))
    return m["base"] * f, used


def backtest(d: pd.DataFrame) -> dict:
    """Leave-one-out: for each mature Short, predict from the others. Compares factors vs baseline-only."""
    errs_f, errs_b, inside = [], [], []
    for i in d.index:
        train = d.drop(index=i)
        m = _model(train)
        actual = float(d.loc[i, "views"])
        if actual <= 0 or m["base"] <= 0:
            continue
        pf, _ = _predict(m, _features(d.loc[i].to_dict()), True)
        pb, _ = _predict(m, {}, False)
        errs_f.append(abs(np.log(max(pf, 1) / actual)))
        errs_b.append(abs(np.log(max(pb, 1) / actual)))
        inside.append(pb * m["q25"] <= actual <= pb * m["q75"])
    if not errs_f:
        return {"n": 0}
    return {"n": len(errs_f), "median_abs_pct_error_baseline": float(np.expm1(np.median(errs_b))),
            "median_abs_pct_error_with_factors": float(np.expm1(np.median(errs_f))),
            "share_inside_range": float(np.mean(inside)), "factors_help": bool(np.median(errs_f) < np.median(errs_b) * 0.97)}


def predict_short(results: dict, title: str, duration_seconds: float, weekday: int | None = None, topic: str | None = None) -> dict:
    """Estimate typical views at maturity (about 14+ days) for a planned Short. weekday: 0=Mon..6=Sun (UTC) or None."""
    df = results["videos"]
    shorts = df[(df["format"] == SHORT) & (df["age_days"] >= MATURE_DAYS)]
    known = [t for t in results["topics"]["topic"].tolist() if not str(t).startswith("Other")] if len(results.get("topics", [])) else []
    notes = ["Estimate of typical views at maturity (14+ days), based on your own public Short history. Not a forecast and not a guarantee."]
    n = len(shorts)
    out: dict = {"available": False, "n_shorts": n, "notes": notes}
    if n >= 3:
        d = _prep(shorts, known)
        m = _model(d)
        bt = backtest(d) if n >= MIN_FOR_MODEL else {"n": 0}
        use_factors = bool(bt.get("factors_help"))
        feats = {"topic": topic or _topic_of(title, known), "pattern": classify_pattern(title),
                 "weekday": weekday if weekday is not None else -1, "length": length_bucket(duration_seconds)}
        est, used = _predict(m, feats, use_factors)
        confidence = "High" if n >= 25 else "Medium" if n >= 12 else "Low"
        if bt.get("n", 0) and not use_factors:
            notes.append("Back-test: topic/title/weekday/length adjustments did NOT beat your plain Shorts median, so the estimate uses the median only.")
        if n < MIN_FOR_MODEL:
            notes.append(f"Only {n} mature Shorts - using their median with a wide range (adjustments need {MIN_FOR_MODEL}+).")
        out.update(available=True, method="Your Shorts history", baseline=m["base"], estimate_mid=est, estimate_low=est * m["q25"],
                   estimate_high=est * m["q75"], factors_used=use_factors, factors=pd.DataFrame(
                       [{"feature": k, "value": str(v[0]), "factor": round(v[1], 2), "similar_shorts": v[2]} for k, v in used.items()]),
                   backtest=bt, confidence=confidence)
        out["range_note"] = "Range = how widely your own Shorts' views spread (25th-75th percentile of your Shorts relative to their median); about half of results fall inside, and some fall outside."
        return out
    # fallback: size-adjusted peers
    subs = results["channel"].get("subscribers")
    peer_rates = []
    for name, cdf in results.get("competitor_videos", {}).items():
        ch = results.get("competitor_channels", {}).get(name, {})
        ps = cdf[(cdf["format"] == SHORT) & (cdf["age_days"] >= MATURE_DAYS)]
        if len(ps) >= 5 and ch.get("subscribers"):
            peer_rates.append(float(ps["views"].median()) / ch["subscribers"])
    if subs and len(peer_rates) >= 3:
        base = float(np.median(peer_rates)) * subs
        out.update(available=True, method="Peers' Shorts scaled to your subscriber count", baseline=base, estimate_mid=base,
                   estimate_low=base * 0.25, estimate_high=base * 3.0, factors_used=False, factors=pd.DataFrame(), backtest={"n": 0}, confidence="Low",
                   range_note="Very wide range: you have too few Shorts of your own, so this uses similar channels' Shorts scaled by audience size.")
        notes.append(f"You have {n} mature Short(s) (3 needed). Peer-based estimate shown instead; treat it as a rough starting point only.")
    else:
        out["reason"] = f"Needs at least 3 of your Shorts that are {MATURE_DAYS}+ days old (you have {n}), or 3+ peers with Shorts and a public subscriber count."
    return out
