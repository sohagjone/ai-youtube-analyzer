"""Upload planner: best publishing slots and a short upload schedule from your channel's own history.

Slot effect = median (views / your median) of mature videos published in that weekday + 3-hour block (in your time zone),
shrunk toward a prior so thin slots cannot dominate. The prior is 'no effect', nudged half-way toward what your discovered
peers show for the same slot (their audiences differ, so it counts only half). It is an observed pattern - not a guarantee
that publishing then will raise views; with small samples treat it as a test.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHRINK_K = 3.0
OWN_BONUS = 0.3
PEER_K = 10.0
MIN_AGE_DAYS = 14
COMMON_TZ = ["UTC", "Asia/Dhaka", "Asia/Kolkata", "Asia/Dubai", "Asia/Singapore", "Asia/Tokyo", "Europe/London", "Europe/Paris",
             "Europe/Istanbul", "Africa/Lagos", "America/New_York", "America/Chicago", "America/Los_Angeles", "America/Sao_Paulo", "Australia/Sydney"]


def tz_of(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - unknown name or missing tz database
        return ZoneInfo("UTC")


def _block_label(h: int) -> str:
    return f"{h:02d}:00-{(h + 3) % 24:02d}:00"


def _rel_by_slot(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    d = df[df["age_days"] >= MIN_AGE_DAYS]
    if len(d) < 3:
        return pd.DataFrame(columns=["weekday", "block", "n", "rel"])
    local = d["published_at"].dt.tz_convert(tz_of(tz))
    med = float(d["views"].median())
    x = pd.DataFrame({"weekday": local.dt.dayofweek, "block": (local.dt.hour // 3) * 3, "rel": d["views"] / med if med > 0 else 1.0})
    return x.groupby(["weekday", "block"])["rel"].agg(n="count", rel="median").reset_index()


def slot_table(df: pd.DataFrame, tz: str = "UTC", peers: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    own = _rel_by_slot(df, tz)
    peer_frames = [_rel_by_slot(p, tz) for p in (peers or {}).values()]
    peer_frames = [p for p in peer_frames if len(p)]
    if peer_frames:
        pooled = pd.concat(peer_frames).groupby(["weekday", "block"]).apply(
            lambda g: pd.Series({"peer_n": g["n"].sum(), "peer_rel": float(np.average(g["rel"], weights=g["n"]))}), include_groups=False).reset_index()
    else:
        pooled = pd.DataFrame(columns=["weekday", "block", "peer_n", "peer_rel"])
    grid = pd.MultiIndex.from_product([range(7), range(0, 24, 3)], names=["weekday", "block"]).to_frame(index=False)
    t = grid.merge(own, how="left", on=["weekday", "block"]).merge(pooled, how="left", on=["weekday", "block"])
    t["n"] = t["n"].fillna(0).astype(int)
    t["peer_n"] = t["peer_n"].fillna(0).astype(int)
    peer_shrunk = (t["peer_n"] * t["peer_rel"].fillna(1.0) + PEER_K * 1.0) / (t["peer_n"] + PEER_K)  # peers' thin slots are noisy too
    prior = np.where(t["peer_n"] >= 5, 0.5 * (1 + peer_shrunk), 1.0)
    own_rel = t["rel"].fillna(1.0)
    t["score"] = (t["n"] * own_rel + SHRINK_K * prior) / (t["n"] + SHRINK_K)
    t["day"] = t["weekday"].map(dict(enumerate(WEEKDAYS)))
    t["window"] = t["block"].apply(_block_label)
    t["evidence"] = np.where(t["n"] >= 2, "own history", np.where(t["peer_n"] >= 5, "peers only", "none"))
    t["priority"] = t["score"] + np.where(t["evidence"] == "own history", OWN_BONUS, 0.0)  # slots backed by your own videos are preferred
    return t[["weekday", "day", "block", "window", "n", "rel", "peer_n", "peer_rel", "score", "evidence", "priority"]].rename(
        columns={"n": "own_videos", "rel": "own_median_rel"})


def reliability(df: pd.DataFrame) -> tuple[str, str]:
    n = int((df["age_days"] >= MIN_AGE_DAYS).sum())
    if n >= 30:
        return "Moderate", f"{n} mature videos - patterns are indicative, still not proof."
    if n >= 15:
        return "Low-moderate", f"{n} mature videos - use as a test, not a rule."
    return "Low", f"Only {n} mature videos - treat every slot as an experiment."


def recommend(slots: pd.DataFrame, n: int = 3) -> dict:
    ok = slots[slots["evidence"] != "none"].sort_values("priority", ascending=False)
    avoid = slots[slots["own_videos"] >= 3].sort_values("score").head(2)
    return {"best": ok.head(n).reset_index(drop=True), "avoid": avoid[avoid["score"] < 0.9].reset_index(drop=True)}


def plan_uploads(slots: pd.DataFrame, uploads_per_30_days: float, ideas: list[tuple[str, str]], now_local: datetime,
                 horizon_days: int = 14, lead_hours: int = 12) -> pd.DataFrame:
    """Schedule uploads over the horizon on the best-scoring weekday slots, at most one per day."""
    per_week = float(np.clip(uploads_per_30_days / 30 * 7, 1, 7))
    count = int(np.clip(round(per_week * horizon_days / 7), 1, horizon_days))
    usable = slots[slots["evidence"] != "none"]
    if usable.empty:
        usable = slots
    best_by_day = usable.sort_values("priority", ascending=False).drop_duplicates("weekday").set_index("weekday")
    earliest = now_local + timedelta(hours=lead_hours)
    days = []
    for i in range(horizon_days + 1):
        d = (now_local + timedelta(days=i)).replace(minute=0, second=0, microsecond=0)
        wd = d.weekday()
        if wd not in best_by_day.index:
            continue
        row = best_by_day.loc[wd]
        when = d.replace(hour=int(row["block"]))
        if when < earliest:
            continue
        days.append((when, float(row["priority"]), int(row["own_videos"]), row["evidence"], float(row["score"])))
    chosen: list[tuple] = []
    for item in sorted(days, key=lambda x: -x[1]):
        if len(chosen) >= count:
            break
        if all(abs((item[0] - c[0]).total_seconds()) >= 20 * 3600 for c in chosen):
            chosen.append(item)
    chosen.sort(key=lambda x: x[0])
    rows = []
    for k, (when, _prio, n, ev, score) in enumerate(chosen):
        title, fmt = ideas[k % len(ideas)] if ideas else ("[idea]", "Long-form")
        rows.append({"publish_at_local": when.strftime("%Y-%m-%d %H:%M"), "day": WEEKDAYS[when.weekday()], "idea": title, "format": fmt,
                     "slot_score": round(score, 2), "own_videos_in_slot": n, "evidence": ev,
                     "note": "Observed slot effect; not a guarantee." if ev == "own history" else "Little or no data for this slot - treat as a test."})
    return pd.DataFrame(rows)
