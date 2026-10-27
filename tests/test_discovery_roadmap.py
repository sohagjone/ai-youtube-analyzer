from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import competitor_discovery as disc
from src import roadmap as rm
from src.config import is_truthy, read_setting, resolve_data_mode
from src.pipeline import refresh_roadmap, run_analysis
from src.sample_data import SampleClient
from src.youtube_api import QuotaExceededError, YouTubeClient

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


# ------------------------------------------------------------------ config
def test_data_mode_resolution():
    assert resolve_data_mode("AIza123") == "live"
    assert resolve_data_mode("") == "sample" and resolve_data_mode(None) == "sample" and resolve_data_mode("  ") == "sample"
    assert resolve_data_mode("AIza123", force_sample=True) == "sample"
    assert is_truthy("1") and is_truthy("True") and not is_truthy("0") and not is_truthy(None)


def test_read_setting_prefers_secrets_then_env(monkeypatch):
    monkeypatch.setenv("X_SETTING", "from-env")
    assert read_setting("X_SETTING", {"X_SETTING": "from-secrets"}) == "from-secrets"
    assert read_setting("X_SETTING", {}) == "from-env"

    class Broken:
        def get(self, k):
            raise RuntimeError("no secrets file")

    assert read_setting("X_SETTING", Broken()) == "from-env"


# ------------------------------------------------------------------ discovery
def test_auto_discovery_finds_at_least_ten_ranked_and_excludes_self(results):
    disc_tbl = results["discovered_competitors"]
    assert len(disc_tbl) >= 10 and len(results["competitors"]) == 1 + len(disc_tbl)
    assert disc_tbl["match_score"].is_monotonic_decreasing
    assert results["channel"]["channel_id"] not in set(disc_tbl["channel_id"])
    assert (disc_tbl["why"].str.len() > 0).all()


def test_discovery_queries_use_own_topics_and_core_keyword(results):
    q = disc.build_queries(results["videos"], results["topics"])
    assert 1 <= len(q) <= 4 and len(set(q)) == len(q)
    assert any("coffee" in x for x in q)  # core keyword comes from the channel's own tags


def test_size_similarity_is_symmetric_and_bounded():
    assert disc._size_similarity(1000, 1000) == 1
    assert disc._size_similarity(1000, 100_000) == 0
    assert disc._size_similarity(None, 5) == 0.5
    assert disc._size_similarity(1000, 10_000) == pytest.approx(disc._size_similarity(10_000, 1000))


def test_discovery_failure_is_a_warning_not_a_crash():
    class QuotaClient(SampleClient):
        def find_channel_candidates(self, *a, **k):
            raise QuotaExceededError("quota gone")

    r = run_analysis(QuotaClient(NOW), "@sample_own", [], sample_size=20, now=NOW, auto_competitors=10)
    assert len(r["videos"]) == 20 and r["competitors"].empty
    assert any("discovery failed" in w for w in r["warnings"])
    assert r["roadmap"]["available"]  # roadmap still produced from planning defaults


def test_live_client_candidate_aggregation_and_topic_channels_excluded():
    class Resp:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def item(cid, title):
        return {"snippet": {"channelId": cid, "channelTitle": title}}

    class Sess:
        def __init__(self):
            self.pages = [{"items": [item("A", "A"), item("A", "A"), item("B", "B - Topic")]}, {"items": [item("A", "A"), item("C", "C")]}]
            self.params = []

        def get(self, url, params=None, timeout=None):
            self.params.append(params)
            return Resp(self.pages.pop(0))

    c = YouTubeClient("AIzaFAKE", session=Sess())
    out = c.find_channel_candidates(["q1", "q2"], published_after=NOW)
    hits = {e["channel_id"]: e["hits"] for e in out}
    assert hits["A"] == 2 and hits["C"] == 1  # a channel counts once per query
    assert all(p["type"] == "video" and "publishedAfter" in p for p in c.session.params)


# ------------------------------------------------------------------ roadmap
def test_kpi_targets_move_toward_benchmark_and_respect_caps(results):
    k = results["roadmap"]["kpis"].dropna(subset=["target_30"])
    assert len(k) >= 4
    for r in k.itertuples():
        assert r.attainment is not None and 0 <= r.attainment <= 1
        seq = [r.you, r.target_30, r.target_60, r.target_90]
        diffs = np.diff(seq)
        assert (diffs >= -1e-9).all() if r.higher_is_better else (diffs <= 1e-9).all()  # monotonic path
        if r.higher_is_better and not r.controllable:
            assert r.target_90 <= r.you * 2 + 1e-9  # outcome KPIs never targeted beyond +100%


def test_controllable_kpis_close_full_gap_by_day_60():
    own = {"uploads_per_30_days": 2.0, "gap_cv": 1.2, "median_views_per_day": 10.0, "median_views_per_1k_subs": 5.0,
           "median_engagement_rate": 0.02, "breakout_rate": 0.1}
    peers = pd.DataFrame([{"uploads_per_30_days": 8, "gap_cv": 0.4, "median_views_per_day": 20, "median_views_per_1k_subs": 10,
                           "median_engagement_rate": 0.04, "breakout_rate": 0.2}] * 5)
    k = rm.build_kpi_table(own, peers).set_index("key")
    assert k.loc["uploads_per_30_days", "target_60"] == pytest.approx(8.0) and k.loc["uploads_per_30_days", "target_30"] == pytest.approx(5.0)
    assert k.loc["gap_cv", "target_60"] == pytest.approx(0.4)
    assert k.loc["median_views_per_day", "target_90"] == pytest.approx(15.0)  # outcome KPI: 50% of the gap by day 90
    assert k.loc["uploads_per_30_days", "attainment"] == pytest.approx(0.25)
    assert k.loc["gap_cv", "attainment"] == pytest.approx(0.4 / 1.2)  # lower-is-better is inverted
    score, comps = rm.health_score(k.reset_index())
    assert 0 < score < 100 and comps["points"].sum() == pytest.approx(score, abs=0.1)


def test_ahead_of_peers_gets_stretch_goal_and_maintain_action():
    own = {"uploads_per_30_days": 12.0, "gap_cv": 0.2, "median_views_per_day": 50.0, "median_views_per_1k_subs": 20.0,
           "median_engagement_rate": 0.08, "breakout_rate": 0.4}
    peers = pd.DataFrame([{"uploads_per_30_days": 6, "gap_cv": 0.5, "median_views_per_day": 20, "median_views_per_1k_subs": 10,
                           "median_engagement_rate": 0.04, "breakout_rate": 0.2}] * 4)
    k = rm.build_kpi_table(own, peers)
    assert (k["attainment"] == 1.0).all()
    score, _ = rm.health_score(k)
    assert score == 100.0
    acts = rm.build_actions(k, {"topic": None, "pattern": None, "weekday": None, "format": None, "breakouts": []})
    assert len(acts) == 1 and "protect" in acts.iloc[0]["action"].lower()


def test_no_peers_uses_labelled_planning_defaults_only():
    own = {"uploads_per_30_days": 2.0, "gap_cv": 1.0, "median_views_per_day": 10.0, "median_views_per_1k_subs": 5.0,
           "median_engagement_rate": 0.02, "breakout_rate": 0.1}
    k = rm.build_kpi_table(own, pd.DataFrame()).set_index("key")
    assert k.loc["uploads_per_30_days", "benchmark_source"].startswith("Planning default")
    assert k.loc["median_views_per_day", "benchmark_source"] == "No benchmark available" and np.isnan(k.loc["median_views_per_day", "target_30"])


def test_daily_plan_is_30_days_with_checkpoints_and_matching_cadence(results):
    plan = results["roadmap"]["daily_plan"]
    assert list(plan["day"]) == list(range(1, 31)) and plan["date"].is_monotonic_increasing
    assert plan.loc[plan["day"].isin([7, 14, 21, 28]), "focus"].eq("Weekly checkpoint").all()
    assert plan.iloc[0]["focus"] == "Baseline" and (plan["est_minutes"] > 0).all() and not plan["done"].any()
    k = results["roadmap"]["kpis"].set_index("key")
    goal = k.loc["uploads_per_30_days", "target_30"]
    publishes = int((plan["focus"] == "Publish").sum())
    assert abs(publishes - goal) <= 2.5  # publish days follow the day-30 cadence goal (rounded, minus checkpoint collisions)
    text = " ".join(plan["task"]).lower()
    assert not any(p in text for p in ("guarantee", "go viral", "will get views"))


def test_scenarios_are_plain_arithmetic_and_labelled():
    kp = pd.DataFrame([{"key": "uploads_per_30_days", "you": 3.0, "target_30": 4.0, "target_60": 5.0, "target_90": 6.0}]).set_index("key", drop=False)
    sc = rm.build_scenarios(kp, 1000.0, 0.10)
    assert sc.loc[0, "videos_in_90_days"] == 9 and sc.loc[0, "views_from_new_videos"] == 9000
    assert sc.loc[1, "videos_in_90_days"] == 15 and sc.loc[2, "assumed_median_views"] == pytest.approx(1100)
    assert "not a forecast" in sc.loc[0, "note"].lower()


def test_momentum_needs_enough_data(results):
    assert rm.momentum(results["videos"].head(3)) == {"available": False}
    assert results["roadmap"]["momentum"]["available"]


def test_refresh_roadmap_uses_new_uplift_and_bumps_version():
    r = run_analysis(SampleClient(NOW), "@sample_own", [], sample_size=30, now=NOW, auto_competitors=10)
    v = r["meta"]["version"]
    refresh_roadmap(r, 0.25)
    assert r["meta"]["version"] == v + 1
    assert r["roadmap"]["scenarios"].iloc[2]["assumed_median_views"] == pytest.approx(float(r["videos"]["views"].median()) * 1.25)


def test_exports_include_roadmap_and_auto_competitor_sheets(results):
    import io

    import openpyxl

    from utils.export_excel import build_workbook
    names = openpyxl.load_workbook(io.BytesIO(build_workbook(results))).sheetnames
    for n in ("Growth Roadmap", "Roadmap Actions", "Daily Plan", "Auto Competitors"):
        assert n in names
