"""Shorts view-range predictor built from THIS channel's own Shorts.

Method: baseline = median views of the channel's settled Shorts; each feature group (title length, number, question, hashtag, duration band,
weekend, topic) gets a shrunk effect from the channel's own history; the effects multiply the baseline. The uncertainty band comes from how
wrong the method was on the channel's past Shorts (leave-one-out). If the features do NOT beat the plain-median baseline in that backtest,
the tool says so and returns the plain baseline range. It predicts views a Short typically reaches after it has settled (about 2 weeks+),
not virality, and cannot see audience retention or impressions (private to the owner).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.audience_profile import SHORT_FMT, extract_hashtags

MIN_SHORTS = 8
FEATURES = ["title_len", "has_number", "has_question", "has_hashtag", "duration_band", "weekend", "topic"]
CAP = (0.6, 1.7)


def _title_len(n: int) -> str:
    return "<=30 chars" if n <= 30 else "31-55 chars" if n <= 55 else ">55 chars"


def _dur_band(s: float) -> str:
    return "<=15s" if s <= 15 else "16-30s" if s <= 30 else "31-45s" if s <= 45 else "46-60s" if s <= 60 else ">60s"


def feature_row(title: str, duration_s: float, publish_dt: pd.Timestamp | None, description: str = "", topic: str = "") -> dict:
    wk = publish_dt.dayofweek >= 5 if publish_dt is not None else False
    return {"title_len": _title_len(len(title)), "has_number": bool(re.search(r"\d", title)),
            "has_question": "?" in title or title.lower().split(" ")[0:1] in (["how"], ["why"], ["what"], ["is"], ["can"]),
            "has_hashtag": bool(extract_hashtags(title, description)), "duration_band": _dur_band(duration_s),
            "weekend": bool(wk), "topic": topic or ""}


def shorts_pool(df: pd.DataFrame, min_age: float = 14) -> pd.DataFrame:
    s = df[df["format"] == SHORT_FMT].copy()
    s = s[s["age_days"] >= min_age] if (s["age_days"] >= min_age).sum() >= MIN_SHORTS else s
    if s.empty:
        return s
    rows = [feature_row(r.title, r.duration_seconds, pd.Timestamp(r.published_at), r.description, getattr(r, "topic", "")) for r in s.itertuples()]
    feats = pd.DataFrame(rows)
    return pd.concat([s.drop(columns=[c for c in feats.columns if c in s.columns]).reset_index(drop=True), feats], axis=1)


def _effects(pool: pd.DataFrame, base: float, k: int = 4) -> dict:
    out: dict[str, dict] = {}
    for f in FEATURES:
        if f not in pool:
            continue
        g = {}
        for grp, sub in pool.groupby(f):
            n = len(sub)
            raw = float(sub["views"].median()) / base if base > 0 else 1.0
            g[grp] = (float(np.clip(1.0 + (raw - 1.0) * n / (n + k), *CAP)), n)
        out[f] = g
    return out


def _apply(effects: dict, row: dict) -> tuple[float, list[dict]]:
    mult, drivers = 1.0, []
    for f, groups in effects.items():
        v = row.get(f)
        if v in groups:
            e, n = groups[v]
            mult *= e
            drivers.append({"feature": f, "value": str(v), "effect": round(e, 2), "history_videos": n})
    return float(np.clip(mult, 0.4, 2.5)), drivers


def predict(df: pd.DataFrame, row: dict) -> dict:
    pool = shorts_pool(df)
    n = len(pool)
    if n < MIN_SHORTS:
        return {"available": False, "n": n, "message": f"Need at least {MIN_SHORTS} Shorts in the analysed sample to predict (found {n}). Raise 'Videos analysed' in the sidebar."}
    views = pool["views"].to_numpy(dtype=float)
    base = float(np.median(views))
    # leave-one-out backtest: does using the features beat the plain median?
    err_m, err_n, ratios = [], [], []
    for i in range(n):
        rest = pool.drop(index=i)
        b = float(rest["views"].median())
        mult, _ = _apply(_effects(rest, b), pool.iloc[i].to_dict())
        pred = max(b * mult, 1.0)
        err_m.append(abs(np.log(max(views[i], 1) / pred)))
        err_n.append(abs(np.log(max(views[i], 1) / max(b, 1))))
        ratios.append(np.log(max(views[i], 1) / pred))
    beats = float(np.median(err_m)) < float(np.median(err_n)) * 0.97
    eff = _effects(pool, base)
    mult, drivers = _apply(eff, row) if beats else (1.0, [])
    center = base * mult
    resid = np.array(ratios if beats else [np.log(max(v, 1) / base) for v in views])
    lo, hi = float(np.exp(np.quantile(resid, 0.2))) * center, float(np.exp(np.quantile(resid, 0.8))) * center
    prob = float((center * np.exp(resid) > base).mean())
    conf = "high" if n >= 40 else "medium" if n >= 20 else "low"
    return {"available": True, "n": n, "baseline_median": base, "multiplier": round(mult, 2), "expected_median": center, "range_low": lo, "range_high": hi,
            "chance_above_channel_median": prob, "features_beat_baseline": beats, "backtest_error_model": float(np.median(err_m)),
            "backtest_error_baseline": float(np.median(err_n)), "confidence": conf, "drivers": drivers,
            "message": ("Your Shorts' features carried measurable signal in a backtest, so they adjust the estimate." if beats else
                        "In a backtest your title/format features did NOT predict your past Shorts better than the plain median, so the range below is the channel's typical Shorts range, not a feature-based estimate."),
            "caveats": ["Estimates settled views (after ~2 weeks), not first-day views or virality.", "Public data excludes retention, impressions and traffic sources; the biggest drivers of Shorts reach are not visible here.",
                        f"Based on {n} of your Shorts; ranges are wide on small samples. It is a planning aid, not a promise."]}
