"""Income estimate, monetization-eligibility timeline and income roadmap.

IMPORTANT - everything here is an ESTIMATE built from public data plus explicit, editable assumptions.
Real revenue, watch hours, Shorts engaged views and monetization status are private to the channel owner
(YouTube Studio > Earn). Nothing here is a forecast or a promise of income.

Method (all assumptions are shown to the user and can be changed):
* Pace      = uploads per 30 days x median views per video, separately for long-form and Shorts (conservative: older
              videos' ongoing views are ignored).
* Watch hrs = long-form views x average duration x assumed % of the video watched (Shorts watch hours do not count).
* Subs gain = monthly views x the channel's lifetime subscribers-per-1,000-views ratio.
* Income    = long-form views x RPM + Shorts views x Shorts RPM, as a low / mid / high range, only once eligible.
* Rules     = YouTube Partner Program thresholds below (ads tier changes on 2027-02-01).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LONG_FMT = "Long-form (>3 min)"
POLICY_CHANGE = pd.Timestamp("2027-02-01", tz="UTC")
REVIEW_LAG_MONTHS = 1  # time between meeting the thresholds and earning (application review)
HORIZON_MONTHS = 24

# (subscribers, qualified public long-form watch hours in 12 months, qualified Shorts views in 90 days)
TIER1 = {"subs": 500, "hours": 3000, "shorts": 3_000_000}
ADS_BEFORE = {"subs": 1000, "hours": 4000, "shorts": 10_000_000}
ADS_AFTER = {"subs": 1000, "hours": 8000, "shorts": 20_000_000}  # new applicants from 2027-02-01
THRESHOLD_SOURCE = "YouTube Partner Program rules as published 2026 (check YouTube Studio > Earn for your own live progress)."

# Illustrative earnings per 1,000 views, USD, creator share: (low, mid, high). Actual RPM depends on niche, audience country,
# season and video length. These are assumptions, not measurements.
NICHE_RPM = {
    "General / mixed": {"long": (1.0, 2.5, 5.0), "short": (0.01, 0.04, 0.08)},
    "Entertainment / gaming / vlog": {"long": (0.5, 1.5, 3.0), "short": (0.01, 0.03, 0.06)},
    "Education / how-to": {"long": (1.5, 4.0, 8.0), "short": (0.01, 0.04, 0.08)},
    "Tech / business / finance": {"long": (3.0, 8.0, 15.0), "short": (0.02, 0.05, 0.10)},
}
STATUS_OPTIONS = {"auto": "Not sure (estimate from public data)", "monetized": "Already monetized", "not_monetized": "Not monetized yet"}

DEFAULT_ASSUMPTIONS = {"status": "auto", "niche": "General / mixed", "view_pct": 0.40, "monthly_growth": 0.0,
                       "packaging_uplift": 0.10, "subs_override": 0, "cadence_ratio": 1.5}


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def ads_threshold(when: pd.Timestamp) -> dict:
    return ADS_AFTER if when >= POLICY_CHANGE else ADS_BEFORE


def _months_ahead(now: pd.Timestamp, m: int) -> pd.Timestamp:
    return now + pd.DateOffset(months=m)


# ----------------------------------------------------------------------------- baseline from public data
def _fmt_stats(d: pd.DataFrame, span_days: float) -> tuple[float, float, float]:
    """(uploads per 30 days, median views of mature videos, mean duration seconds) for one format."""
    if d.empty:
        return 0.0, 0.0, 0.0
    mature = d[d["age_days"] >= 28]
    med = float((mature if len(mature) >= 3 else d)["views"].median())
    return float(len(d) / span_days * 30), med, float(d["duration_seconds"].mean())


def baseline(channel: dict, df: pd.DataFrame, a: dict, now: pd.Timestamp) -> dict:
    created = pd.to_datetime(channel.get("created_at"), utc=True, errors="coerce")
    age_months = max((now - created).days / 30.44, 1.0) if pd.notna(created) else 12.0
    span = max(float((now - df["published_at"].min()).days), 30.0)
    is_long = df["format"] == LONG_FMT
    l_up, l_med, l_dur = _fmt_stats(df[is_long], span)
    s_up, s_med, _ = _fmt_stats(df[~is_long], span)
    l_views, s_views = l_up * l_med, s_up * s_med
    dur_min = l_dur / 60.0
    hours_per_view = dur_min * a["view_pct"] / 60.0
    win12, win3 = min(12.0, age_months), min(3.0, age_months)
    total_views = channel.get("total_views")
    long12 = l_views * win12
    if _ok(total_views) and total_views:
        long12 = min(long12, float(total_views))
    subs = a["subs_override"] if a["subs_override"] else channel.get("subscribers")
    subs_known = _ok(subs) and not channel.get("subscribers_hidden", False) or bool(a["subs_override"])
    subs = float(subs) if _ok(subs) and subs_known else 0.0
    conv = (subs / total_views * 1000.0) if _ok(total_views) and total_views and subs else 0.0
    return {"subs": subs, "subs_known": bool(subs_known), "age_months": age_months, "long_uploads_30": l_up, "short_uploads_30": s_up,
            "long_median_views": l_med, "short_median_views": s_med, "long_views_month": l_views, "short_views_month": s_views,
            "avg_long_minutes": dur_min, "hours_per_view": hours_per_view, "hours_12m": long12 * hours_per_view,
            "window_months": win12, "shorts_90d": s_views * win3, "shorts_window_months": win3, "subs_per_1k_views": conv}


def _rpm(a: dict) -> tuple[tuple, tuple]:
    p = NICHE_RPM.get(a["niche"], NICHE_RPM["General / mixed"])
    return p["long"], p["short"]


def _eligibility_at(subs: float, hours: float, shorts90: float, when: pd.Timestamp) -> dict:
    t = ads_threshold(when)
    by_hours, by_shorts = hours >= t["hours"], shorts90 >= t["shorts"]
    return {"ads": subs >= t["subs"] and (by_hours or by_shorts), "tier1": subs >= TIER1["subs"] and (hours >= TIER1["hours"] or shorts90 >= TIER1["shorts"]),
            "subs_ok": subs >= t["subs"], "hours_ok": by_hours, "shorts_ok": by_shorts}


# ----------------------------------------------------------------------------- projection
def simulate(base: dict, a: dict, name: str, view_mult: float, growth: float, now: pd.Timestamp) -> pd.DataFrame:
    (l_lo, l_mid, l_hi), (s_lo, s_mid, s_hi) = _rpm(a)
    mw = max(int(round(base["window_months"])), 1)
    hist_h = deque([base["hours_12m"] / mw] * mw + [0.0] * (12 - mw), maxlen=12)
    sw = max(int(round(base["shorts_window_months"])), 1)
    hist_s = deque([base["shorts_90d"] / sw] * sw + [0.0] * (3 - sw), maxlen=3)
    subs, rows, start_idx = base["subs"], [], None
    for t in range(0, HORIZON_MONTHS + 1):
        when = _months_ahead(now, t)
        if t > 0:
            f = view_mult * (1 + growth) ** (t - 1)
            lv, sv = base["long_views_month"] * f, base["short_views_month"] * f
            hist_h.append(lv * base["hours_per_view"])
            hist_s.append(sv)
            subs += (lv + sv) * base["subs_per_1k_views"] / 1000.0
        else:
            lv, sv = base["long_views_month"], base["short_views_month"]
        hours, sh90 = float(sum(hist_h)), float(sum(hist_s))
        el = _eligibility_at(subs, hours, sh90, when)
        if el["ads"] and start_idx is None:
            start_idx = t + REVIEW_LAG_MONTHS
        earning = start_idx is not None and t >= start_idx
        inc = lambda r_l, r_s: (lv * r_l + sv * r_s) / 1000.0 if earning else 0.0  # noqa: E731
        rows.append({"scenario": name, "month": t, "date": when.date().isoformat(), "subscribers": round(subs), "watch_hours_12m": round(hours),
                     "shorts_views_90d": round(sh90), "tier1_eligible": el["tier1"], "ads_eligible": el["ads"],
                     "views_per_month": round(lv + sv), "income_low": round(inc(l_lo, s_lo), 2), "income_mid": round(inc(l_mid, s_mid), 2),
                     "income_high": round(inc(l_hi, s_hi), 2)})
    out = pd.DataFrame(rows)
    out["cumulative_mid"] = out["income_mid"].cumsum().round(2)
    return out


def _scenarios(a: dict) -> list[tuple[str, float, float]]:
    ratio = float(np.clip(a["cadence_ratio"], 1.0, 3.0))
    return [("Maintain current pace", 1.0, a["monthly_growth"]),
            ("Reach cadence targets", ratio, a["monthly_growth"]),
            (f"Cadence targets + packaging uplift ({a['packaging_uplift']:+.0%})", ratio * (1 + a["packaging_uplift"]), max(a["monthly_growth"], 0.02))]


def _eligibility_table(proj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in proj.groupby("scenario", sort=False):
        t1, ads = g[g["tier1_eligible"]], g[g["ads_eligible"]]
        first = g[g["income_mid"] > 0]
        last = g.iloc[-1]
        rows.append({"scenario": name,
                     "fan_funding_tier_in_months": int(t1["month"].iloc[0]) if len(t1) else None,
                     "ads_eligible_in_months": int(ads["month"].iloc[0]) if len(ads) else None,
                     "ads_eligible_date": ads["date"].iloc[0] if len(ads) else None,
                     "first_income_date": first["date"].iloc[0] if len(first) else None,
                     f"est_monthly_income_mid_month_{HORIZON_MONTHS}": float(last["income_mid"]),
                     f"est_total_income_mid_{HORIZON_MONTHS}_months": float(last["cumulative_mid"])})
    return pd.DataFrame(rows)


def _reverse_plan(base: dict, a: dict, now: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for n in (6, 12):
        t = ads_threshold(_months_ahead(now, n))
        add_h = max(0.0, t["hours"] - base["hours_12m"])
        views_needed = (add_h / n) / base["hours_per_view"] if base["hours_per_view"] > 0 else np.nan
        med = base["long_median_views"]
        rows.append({"goal": f"Ads eligible in {n} months", "subscribers_needed": max(0, round(t["subs"] - base["subs"])),
                     "subs_per_month_needed": round(max(0, t["subs"] - base["subs"]) / n),
                     "extra_watch_hours_needed": round(add_h), "long_views_per_month_needed": round(views_needed) if _ok(views_needed) else None,
                     "long_videos_per_month_at_median_views": round(views_needed / med, 1) if _ok(views_needed) and med > 0 else None,
                     "threshold_used": f"{t['subs']:,} subs + {t['hours']:,} hours"})
    return pd.DataFrame(rows)


def _steps(base: dict, now: pd.Timestamp, now_el: dict, elig: pd.DataFrame, projections: pd.DataFrame, mid_scn: str) -> pd.DataFrame:
    t = ads_threshold(now)
    g = projections[projections["scenario"] == mid_scn]
    when = lambda col: (g[g[col] > 0]["date"].iloc[0] if (g[col] > 0).any() else "beyond 24 months")  # noqa: E731
    e = elig[elig["scenario"] == mid_scn].iloc[0]
    rows = [
        {"step": 1, "milestone": "Meet account rules", "target": "18+ or an adult AdSense owner, 2-step verification, supported country, no active strikes, original content",
         "today": "Check in YouTube Studio > Earn", "estimated_date": "Now"},
        {"step": 2, "milestone": "Fan-funding tier", "target": f"{TIER1['subs']:,} subs + {TIER1['hours']:,} public watch hours (or {TIER1['shorts'] / 1e6:.0f}M Shorts views / 90 days) + 3 uploads in 90 days",
         "today": "Reached (estimate)" if now_el["tier1"] else f"{base['subs']:,.0f} subs, ~{base['hours_12m']:,.0f} hours (est.)",
         "estimated_date": "Now" if now_el["tier1"] else (_months_ahead(now, int(e["fan_funding_tier_in_months"])).date().isoformat() if _ok(e["fan_funding_tier_in_months"]) else "beyond 24 months")},
        {"step": 3, "milestone": "Ads tier (earn from ads)", "target": f"{t['subs']:,} subs + {t['hours']:,} hours or {t['shorts'] / 1e6:.0f}M Shorts views / 90 days (rises to {ADS_AFTER['hours']:,} hours / {ADS_AFTER['shorts'] / 1e6:.0f}M from 2027-02-01 for new applicants)",
         "today": "Reached (estimate)" if now_el["ads"] else f"{base['subs']:,.0f} subs, ~{base['hours_12m']:,.0f} hours (est.)",
         "estimated_date": "Now" if now_el["ads"] else (e["ads_eligible_date"] or "beyond 24 months")},
        {"step": 4, "milestone": "Apply and pass review", "target": "Apply in YouTube Studio > Earn; review can take about a month", "today": "-", "estimated_date": e["first_income_date"] or "beyond 24 months"},
    ]
    for amount in (100, 500, 1000):
        hit = g[g["income_mid"] >= amount]
        rows.append({"step": len(rows) + 1, "milestone": f"Reach about ${amount:,} per month (mid estimate)", "target": f"${amount:,} / month", "today": "-",
                     "estimated_date": hit["date"].iloc[0] if len(hit) else "beyond 24 months"})
    rows.append({"step": len(rows) + 1, "milestone": "Add non-ad income", "target": "Sponsorships, affiliate links, memberships, products: not estimated here because they depend on your deals and audience",
                 "today": "-", "estimated_date": "After ads tier"})
    return pd.DataFrame(rows)


def build_monetization(channel: dict, df: pd.DataFrame, assumptions: dict | None = None, now: datetime | None = None) -> dict:
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    if df is None or df.empty or "format" not in df:
        return {"available": False}
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts
    base = baseline(channel, df, a, now_ts)
    (l_lo, l_mid, l_hi), (s_lo, s_mid, s_hi) = _rpm(a)
    now_el = _eligibility_at(base["subs"], base["hours_12m"], base["shorts_90d"], now_ts)
    lv, sv = base["long_views_month"], base["short_views_month"]
    potential = {k: (lv * rl + sv * rs) / 1000.0 for k, rl, rs in (("low", l_lo, s_lo), ("mid", l_mid, s_mid), ("high", l_hi, s_hi))}
    status = a["status"]
    earning_now = status == "monetized" or (status == "auto" and now_el["ads"])
    if status == "not_monetized":
        earning_now = False
    scen = _scenarios(a)
    proj = pd.concat([simulate(base, a, n, m, g, now_ts) for n, m, g in scen], ignore_index=True)
    if earning_now:  # already earning: income starts immediately, regardless of the eligibility timer
        for n, m, g in scen:
            idx = proj["scenario"] == n
            for col, (rl, rs) in (("income_low", (l_lo, s_lo)), ("income_mid", (l_mid, s_mid)), ("income_high", (l_hi, s_hi))):
                f = m * (1 + g) ** (proj.loc[idx, "month"] - 1).clip(lower=0)
                f = np.where(proj.loc[idx, "month"] == 0, 1.0, f)
                proj.loc[idx, col] = ((lv * rl + sv * rs) / 1000.0 * f).round(2)
            proj.loc[idx, "cumulative_mid"] = proj.loc[idx, "income_mid"].cumsum().round(2)
            proj.loc[idx, "ads_eligible"] = True
    elig = _eligibility_table(proj)
    if earning_now:
        elig["ads_eligible_in_months"], elig["ads_eligible_date"], elig["first_income_date"] = 0, now_ts.date().isoformat(), now_ts.date().isoformat()
    mid_scn = scen[1][0]
    t_now = ads_threshold(now_ts)
    gaps = {"subs_gap": max(0, t_now["subs"] - base["subs"]), "hours_gap": max(0.0, t_now["hours"] - base["hours_12m"]),
            "shorts_gap": max(0.0, t_now["shorts"] - base["shorts_90d"])}
    notes = [
        "These are ESTIMATES from public data and editable assumptions. Real income, watch hours, Shorts engaged views and monetization status are only visible to the channel owner in YouTube Studio.",
        "Watch hours = long-form views x average duration x the % of each video you assume is watched. Shorts watch hours do not count toward the hours requirement; Shorts qualify only through the separate Shorts-views route.",
        f"RPM (income per 1,000 views) is an assumption for '{a['niche']}': long-form ${l_lo:g}-${l_hi:g}, Shorts ${s_lo:g}-${s_hi:g}. Your real RPM depends on audience country, niche, season and ad demand.",
        "Pace uses uploads per 30 days x median views per video and ignores ongoing views on older videos, so it leans conservative for established channels and optimistic for channels whose recent videos are weaker than older ones.",
        "Subscriber growth uses this channel's lifetime subscribers per 1,000 views; real growth can be very different.",
        f"From 2027-02-01 new applicants for ads need {ADS_AFTER['hours']:,} watch hours or {ADS_AFTER['shorts'] / 1e6:.0f}M Shorts views; the timeline applies whichever rule is in force in each month. {THRESHOLD_SOURCE}",
    ]
    if not base["subs_known"]:
        notes.insert(0, "Subscribers are hidden for this channel, so the timeline uses 0. Enter your real subscriber count in the assumptions to get a usable timeline.")
    return {"available": True, "assumptions": a, "baseline": base, "status": status, "earning_now": bool(earning_now), "eligible_now": now_el,
            "potential_monthly_income": potential, "gaps": gaps, "projection": proj, "eligibility": elig,
            "reverse_plan": _reverse_plan(base, a, now_ts), "steps": _steps(base, now_ts, now_el, elig, proj, mid_scn),
            "rpm": {"long": (l_lo, l_mid, l_hi), "short": (s_lo, s_mid, s_hi)}, "notes": notes}
