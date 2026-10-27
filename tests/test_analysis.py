import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import channel_analysis as ca
from src import competitor_analysis as comp_a
from src import content_gap_analysis as gaps_a
from src import content_strategy as strat
from src import title_analysis as ta
from src import topic_analysis as topa
from src import video_analysis as va
from src.pipeline import apply_manual_topics, recompute_topics, run_analysis
from src.sample_data import SampleClient
from utils.data_processing import videos_to_df

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


def make_df(views, days_ago=None, titles=None, durations=None, likes=None, comments=None):
    n = len(views)
    days_ago = days_ago or [10 * (i + 1) for i in range(n)]
    rows = []
    for i in range(n):
        rows.append({"video_id": f"v{i}", "title": (titles or [f"Video number {i}"] * n)[i], "url": f"https://youtu.be/v{i}",
                     "published_at": (NOW - timedelta(days=days_ago[i])).isoformat(), "duration_seconds": (durations or [600] * n)[i],
                     "views": views[i], "likes": (likes or [None] * n)[i], "comments": (comments or [None] * n)[i],
                     "description": "", "tags": [], "thumbnail_url": None, "category": None, "channel_id": "c", "channel_title": "Chan"})
    return va.enrich_videos(videos_to_df(rows), NOW)


# ------------------------------------------------------------------ calculations
def test_mean_median_and_relative_performance():
    df = make_df([100, 200, 300, 1000])
    s = va.summarize_videos(df)
    assert s["mean_views"] == 400 and s["median_views"] == 250
    rel = dict(zip(df["video_id"], df["relative_to_median"]))
    assert rel["v3"] == pytest.approx(4.0) and rel["v0"] == pytest.approx(0.4)


def test_video_age_and_views_per_day_floor():
    df = make_df([100, 100], days_ago=[0, 50])  # published 'now' -> age floored to 1 day, no division by zero
    row0 = df[df["video_id"] == "v0"].iloc[0]
    assert row0["age_days"] == 1 and row0["views_per_day"] == 100
    assert df[df["video_id"] == "v1"].iloc[0]["views_per_day"] == pytest.approx(2.0)


def test_below_typical_requires_age_and_new_videos_not_judged():
    df = make_df([1000, 1000, 1000, 100, 100], days_ago=[100, 200, 300, 400, 5])
    flags = dict(zip(df["video_id"], df["performance_flag"]))
    assert flags["v3"] == "Below typical" and flags["v4"] == "Too new to judge"


def test_engagement_handles_hidden_likes_and_comments():
    df = make_df([1000, 1000, 1000], likes=[50, None, None], comments=[10, 5, None])
    e = df.set_index("video_id")["engagement_rate"]
    assert e["v0"] == pytest.approx(0.06) and e["v1"] == pytest.approx(0.005) and math.isnan(e["v2"])


def test_fewer_videos_than_requested_and_empty_df():
    df = make_df([5, 7])
    assert va.summarize_videos(df)["videos_in_sample"] == 2
    empty = va.enrich_videos(videos_to_df([]), NOW)
    assert va.summarize_videos(empty) == {"videos_in_sample": 0}
    assert ca.upload_frequency(empty)["uploads_per_30_days"] is None


def test_date_filter_inclusive():
    df = make_df([1, 2, 3, 4], days_ago=[5, 15, 25, 35])
    out = va.filter_by_date(df, NOW - timedelta(days=30), NOW - timedelta(days=10))
    assert sorted(out["video_id"]) == ["v1", "v2"]


def test_lifetime_and_sample_stats_are_separate():
    df = make_df([100, 300])
    ch = {"title": "C", "total_views": 1_000_000, "total_videos": 100, "subscribers": 10, "created_at": "2020-01-01T00:00:00Z"}
    s = ca.summarize_channel(ch, df)
    assert s["lifetime"]["lifetime_avg_views_per_video"] == 10_000 and s["sample"]["median_views"] == 200
    scopes = set(ca.overview_table(s)["Scope"])
    assert any(x.startswith("Lifetime") for x in scopes) and any(x.startswith("Selected sample") for x in scopes)


# ------------------------------------------------------------------ titles
def test_title_features_and_flags():
    df = make_df([10, 20, 30], titles=["5 Tips for Better Espresso", "How to grind coffee?", "SHOCKING!! You won't believe THIS secret"])
    out = ta.analyze_titles(df).set_index("video_id")
    assert out.loc["v0", "title_has_number"] and out.loc["v0", "title_pattern"] == "Number / list"
    assert out.loc["v1", "title_has_question"] and out.loc["v1", "title_pattern"] == "How-to / guide"
    assert "sensational" in out.loc["v2", "wording_flags"] and out.loc["v0", "wording_flags"] == ""


def test_title_suggestions_never_return_original_and_are_capped():
    ideas = ta.suggest_titles_rule_based("INSANE ESPRESSO SECRETS!!! 🔥", ["espresso"])
    assert 1 <= len(ideas) <= 3
    assert all(i["title"].lower() != "INSANE ESPRESSO SECRETS!!! 🔥".lower() and i["improvement"] for i in ideas)


def test_similar_titles_detected():
    df = make_df([1, 2, 3], titles=["How to make cold brew at home", "How to make cold brew at home fast", "Totally unrelated guitar lesson"])
    pairs = ta.similar_titles(df)
    assert len(pairs) == 1 and "guitar" not in pairs.iloc[0]["title_a"].lower()


def test_ai_title_suggestion_uses_provider_and_fails_loudly():
    class Good:
        def complete_json(self, *a, **k):
            return [{"title": "A", "improvement": "x"}, {"title": "B", "improvement": "y"}, {"title": "C", "improvement": "z"}, {"title": "D"}]

    class Bad:
        def complete_json(self, *a, **k):
            return []

    assert len(ta.suggest_titles_ai(Good(), "t")) == 3
    with pytest.raises(ValueError):
        ta.suggest_titles_ai(Bad(), "t")


def test_thumbnail_stats_on_synthetic_image():
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (64, 36), (255, 255, 255)).save(buf, format="PNG")
    stats = ta.thumbnail_stats(buf.getvalue())
    assert stats["brightness"] == pytest.approx(1.0, abs=0.01) and stats["contrast"] == 0 and stats["visual_busyness"] == 0


# ------------------------------------------------------------------ topics
def test_auto_topics_group_by_shared_terms():
    titles = ["Espresso basics", "Espresso grinder settings", "Espresso shot timing", "Guitar chords for beginners",
              "Guitar tuning tips", "Guitar strumming patterns", "Random vlog"]
    df = make_df([100] * 7, titles=titles)
    t = topa.auto_topics(df)
    assert t.iloc[0] == t.iloc[1] == t.iloc[2] == "Espresso" and t.iloc[3] == t.iloc[4] == t.iloc[5] == "Guitar"
    assert t.iloc[6] == topa.OTHER


def test_custom_topics_and_mapping_parser():
    mapping = topa.parse_topic_mapping("Gear: grinder, kettle\nBrewing: pour over, espresso\nbad line")
    assert mapping == {"Gear": ["grinder", "kettle"], "Brewing": ["pour over", "espresso"]}
    df = make_df([1, 2, 3], titles=["Best grinder", "Pour over guide", "Weekly vlog"])
    assert list(topa.custom_topics(df, mapping)) == ["Gear", "Brewing", topa.OTHER]


def test_topic_summary_observations():
    titles = ["Espresso a", "Espresso b", "Espresso c", "Guitar a", "Guitar b", "Guitar c"]
    df = topa.assign_topics(make_df([900, 1000, 1100, 100, 110, 120], titles=titles))
    summ = topa.topic_summary(df).set_index("topic")
    assert "Strong" in summ.loc["Espresso", "observation"] and "Weaker" in summ.loc["Guitar", "observation"]


# ------------------------------------------------------------------ competitors & gaps
@pytest.fixture(scope="module")
def sample_results():
    return run_analysis(SampleClient(NOW), "@sample_own", ["@rival_a", "@rival_b"], sample_size=40, now=NOW)


def test_competitor_comparison_has_you_first_and_no_invented_values(sample_results):
    t = sample_results["competitors"]
    assert list(t["role"]) == ["You", "Competitor", "Competitor"]
    assert t["median_views"].notna().all() and (t["videos_in_sample"] == 40).all()


def test_competitor_limit_duplicates_and_self_skipped():
    r = run_analysis(SampleClient(NOW), "@sample_own", ["@rival_a", "@RIVAL_A", "@sample_own", "bad url here"], sample_size=15, now=NOW)
    assert len(r["competitor_videos"]) == 1
    assert any("same channel" in w for w in r["warnings"]) and any("bad url here" in w for w in r["warnings"])


def test_competitor_failure_does_not_abort_analysis():
    class Flaky(SampleClient):
        def fetch_channel_bundle(self, ref, *a, **k):
            if "rival" in ref.value:
                from src.youtube_api import ChannelNotFoundError
                raise ChannelNotFoundError("nope")
            return super().fetch_channel_bundle(ref, *a, **k)

    r = run_analysis(Flaky(NOW), "@sample_own", ["@rival_a"], sample_size=15, now=NOW)
    assert r["competitors"].empty and any("skipped" in w for w in r["warnings"]) and len(r["videos"]) == 15


def test_gap_scores_are_transparent_and_bounded(sample_results):
    g = sample_results["gaps"]
    assert len(g) > 0 and g["priority_score"].between(0, 100).all() and g["confidence"].isin(["High", "Medium", "Low"]).all()
    row = g.iloc[0]
    expected = gaps_a.priority_score(row["relevance"], row["demand_evidence"], row["differentiation"], row["feasibility"])
    assert row["priority_score"] == pytest.approx(expected, abs=0.6)  # components are rounded to 2 dp in the table
    assert (g["priority_score"].diff().dropna() <= 0).all()  # ranked descending
    assert g["limitations"].str.contains("search").all()  # never implies search-volume data


def test_gap_topics_are_ones_you_rarely_cover(sample_results):
    own_titles = " ".join(sample_results["videos"]["title"]).lower()
    comp_gaps = sample_results["gaps"][sample_results["gaps"]["gap_type"] == "Competitor topic gap"]
    assert len(comp_gaps) > 0
    assert any(t.lower() not in own_titles for t in comp_gaps["topic"])


def test_gaps_work_without_competitors():
    r = run_analysis(SampleClient(NOW), "@sample_own", [], sample_size=40, now=NOW)
    assert (r["gaps"]["gap_type"] != "Competitor topic gap").all()


def test_recompute_and_manual_topic_corrections_bump_version():
    r = run_analysis(SampleClient(NOW), "@sample_own", ["@rival_a"], sample_size=30, now=NOW)
    v0 = r["meta"]["version"]
    recompute_topics(r, {"Everything": ["coffee", "espresso", "brew", "pour", "grinder"]})
    assert r["meta"]["version"] == v0 + 1 and "Everything" in set(r["videos"]["topic"])
    vid = r["videos"]["video_id"].iloc[0]
    apply_manual_topics(r, {vid: "Hand Picked"})
    assert r["videos"].set_index("video_id").loc[vid, "topic"] == "Hand Picked" and "Hand Picked" in set(r["topics"]["topic"])


# ------------------------------------------------------------------ content plan
def test_calendar_length_and_evidence_labels(sample_results):
    for days, ppw, expected in [(7, 3, 3), (30, 3, 13), (7, 7, 7)]:
        cal = strat.generate_calendar(sample_results["gaps"], sample_results["topics"], days=days, posts_per_week=ppw, start=NOW.date())
        assert len(cal) == expected and cal["date"].is_monotonic_increasing
        assert cal["basis"].str.startswith(("Evidence", "Creative hypothesis")).all()


def test_calendar_without_data_is_labelled_hypothesis_and_has_no_promises():
    cal = strat.generate_calendar(pd.DataFrame(), None, days=7, posts_per_week=2, niche="woodworking", start=NOW.date())
    assert (cal["basis"] == "Creative hypothesis (no channel data behind this idea)").all()
    text = " ".join(cal.astype(str).values.ravel()).lower()
    assert not any(p in text for p in ("guarantee", "go viral", "will get views"))


def test_calendar_ai_failure_falls_back_to_rule_based(sample_results):
    class Broken:
        is_configured = True

        def complete_json(self, *a, **k):
            raise RuntimeError("boom")

    cal = strat.generate_calendar(sample_results["gaps"], None, days=7, posts_per_week=3, provider=Broken(), start=NOW.date())
    assert len(cal) == 3 and (cal["source"] == "Rule-based").all()
