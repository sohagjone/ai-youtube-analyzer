import pandas as pd

from src import monetization as mon
from src.pipeline import refresh_monetization


def test_monetization_is_built(results):
    m = results["monetization"]
    assert m["available"]
    assert len(m["projection"]) == 3 * (mon.HORIZON_MONTHS + 1)
    assert {"low", "mid", "high"} <= set(m["potential_monthly_income"])
    assert m["potential_monthly_income"]["low"] <= m["potential_monthly_income"]["mid"] <= m["potential_monthly_income"]["high"]


def test_not_monetized_means_zero_income_now(results):
    m = refresh_monetization(results, {"status": "not_monetized"})["monetization"]
    assert not m["earning_now"]
    assert m["projection"].query("month == 0")["income_mid"].eq(0).all()
    refresh_monetization(results, {"status": "auto"})


def test_monetized_status_gives_income_now(results):
    m = refresh_monetization(results, {"status": "monetized"})["monetization"]
    assert m["earning_now"] and m["projection"].query("month == 0")["income_mid"].gt(0).all()
    refresh_monetization(results, {"status": "auto"})


def test_small_channel_needs_time_and_thresholds_rise_in_2027():
    now = pd.Timestamp("2026-10-27", tz="UTC")
    assert mon.ads_threshold(now)["hours"] == 4000
    assert mon.ads_threshold(pd.Timestamp("2027-02-01", tz="UTC"))["hours"] == 8000
    n = 40
    df = pd.DataFrame({"video_id": [str(i) for i in range(n)], "published_at": [now - pd.Timedelta(days=7 * i + 1) for i in range(n)],
                       "views": [300] * n, "duration_seconds": [600] * n, "format": ["Long-form (>3 min)"] * n, "age_days": [7 * i + 1 for i in range(n)]})
    ch = {"subscribers": 120, "total_views": 12000, "created_at": (now - pd.Timedelta(days=400)).isoformat(), "subscribers_hidden": False}
    m = mon.build_monetization(ch, df, {"cadence_ratio": 1.0}, now.to_pydatetime())
    assert not m["eligible_now"]["ads"] and not m["earning_now"]
    assert m["gaps"]["subs_gap"] == 880
    assert m["projection"]["income_mid"].max() == 0  # tiny channel never qualifies within 24 months at this pace
    assert m["eligibility"]["ads_eligible_in_months"].isna().all()


def test_hidden_subscribers_flagged_and_override_works():
    now = pd.Timestamp("2026-10-27", tz="UTC")
    n = 20
    df = pd.DataFrame({"published_at": [now - pd.Timedelta(days=10 * i + 1) for i in range(n)], "views": [5000] * n, "duration_seconds": [900] * n,
                       "format": ["Long-form (>3 min)"] * n, "age_days": [10 * i + 1 for i in range(n)]})
    ch = {"subscribers": None, "subscribers_hidden": True, "total_views": 400000, "created_at": (now - pd.Timedelta(days=900)).isoformat()}
    assert not mon.build_monetization(ch, df, None, now.to_pydatetime())["baseline"]["subs_known"]
    m = mon.build_monetization(ch, df, {"subs_override": 8000}, now.to_pydatetime())
    assert m["baseline"]["subs_known"] and m["baseline"]["subs"] == 8000
