import io
from datetime import date

import pandas as pd
import pytest

from src import audience_profile as ap
from src import copyright_risk as cr
from src import editing_advice as ea
from src import media_analysis as ma
from src import metadata_generator as mg
from src import shorts_predictor as sp
from src import trending as tr
from src import upload_planner as up
from src.sample_data import SampleClient


@pytest.fixture(scope="module")
def df(results):
    return results["videos"]


@pytest.fixture(scope="module")
def prof(df):
    return ap.build_profile(df)


def test_profile(prof, results):
    assert prof["available"] and prof["n_videos"] == len(results["videos"])
    assert results["audience_profile"]["available"]


def test_titles_never_invent_numbers_and_hashtags_are_few(prof):
    g = mg.generate(prof, "home espresso", ["espresso"], ["one"], "long", 600)
    assert not g["titles"]["title"].str.contains(r"^\d").any()  # a single key point must not produce a "N things" title
    assert not any(t.startswith("1 ") for t in mg.candidate_titles("home espresso", ["one"], "long", 600))
    assert any(t.startswith("3 things") for t in mg.candidate_titles("home espresso", ["a", "b", "c"], "long", 600))
    g2 = mg.generate(prof, "home espresso", ["espresso", "grinder"], ["a", "b", "c"], "long", 600)
    assert len(g2["hashtags"]) <= 5 and g2["titles"]["channel_fit_score"].between(0, 100).all()
    with pytest.raises(ValueError):
        mg.generate(prof, "  ")


def test_ai_failure_falls_back_to_rules(prof):
    class Broken:
        is_configured = True

        def complete_json(self, *a, **k):
            raise RuntimeError("down")

    g = mg.generate(prof, "latte art", provider=Broken())
    assert len(g["titles"]) and "AI unavailable" in g["note"]


def test_shorts_predictor_range_and_small_sample(df):
    row = sp.feature_row("How to dial in espresso?", 30, pd.Timestamp.now(tz="UTC"), "", "")
    p = sp.predict(df, row)
    if p["available"]:
        assert p["range_low"] <= p["expected_median"] * 1.0001 and p["expected_median"] <= p["range_high"] * 1.0001
        assert 0 <= p["chance_above_channel_median"] <= 1 and p["caveats"]
    assert not sp.predict(df[df["format"] != ap.SHORT_FMT], row)["available"]


def test_planner_assigns_unique_days_on_best_weekdays(df, results):
    plan = up.build_plan(df, results["gaps"], results["topics"], start=date(2026, 11, 2), days=14, posts_per_week=3)
    assert len(plan) >= 1 and plan["date"].is_unique
    assert pd.to_datetime(plan["date"]).between("2026-11-02", "2026-11-15").all()
    assert (pd.to_datetime(plan["date"]).dt.day_name() == plan["weekday"]).all()


def test_copyright_levels(df):
    risky = cr.assess("Full movie lyrics compilation", "no copyright intended", [], music="Popular commercial song", clips="Several clips / a compilation")
    clean = cr.assess("My original tutorial", "", [], music="None / only my own voice and sounds", clips="None", images="My own", voice="My own voice or face")
    assert risky["level"] == "High" and clean["level"] == "Low" and clean["score"] == 0
    assert "Content ID" in clean["findings"]["finding"].str.cat() and "not legal advice" in clean["disclaimer"]
    meta = cr.assess("t", media={"available": True, "metadata_tags": {"artist": "Some Band"}}, music="None / only my own voice and sounds")
    assert (meta["findings"]["area"] == "File metadata").any()


def test_trending_scoring_and_import():
    c = SampleClient()
    t = tr.trending(c, "US", None, 50_000)
    assert len(t) and t["potential_score"].between(0, 100).all()
    out, problems = tr.creator_outliers(c, ["@rival_a"], 20)
    assert len(out) and not problems and (out["outlier_x"] > 0).all()
    ex, notes = tr.import_external(pd.read_csv(io.StringIO(tr.TEMPLATE_CSV)), 10_000)
    assert len(ex) == 1 and ex.iloc[0]["platform"] == "TikTok"
    bad, notes = tr.import_external(pd.DataFrame({"title": ["x"], "views": ["abc"], "posted_at": ["2026-01-01"]}))
    assert bad.empty and notes
    assert "Missing required" in tr.import_external(pd.DataFrame({"a": [1]}))[1][0]


def test_media_guards(monkeypatch):
    assert "Unsupported file type" in ma.analyze_upload(b"x", "virus.exe")["notes"][0]
    monkeypatch.setattr(ma, "ffmpeg_available", lambda: False)
    assert not ma.analyze_upload(b"x", "a.mp4")["available"]


def test_editing_advice_uses_measurements(prof):
    media = {"available": True, "loudness_lufs": -24.0, "true_peak_dbfs": 0.5, "silent_gaps": 5, "vertical": False, "width": 1920, "height": 1080, "has_video": True,
             "cuts_per_minute": 2.0, "duration_seconds": 30}
    adv = ea.build_advice("short", 30, "Upbeat & energetic", prof, media)
    areas = set(adv["area"])
    assert {"Loudness", "Audio peaks", "Dead air", "Frame", "Pacing"} <= areas
    plan = ea.music_plan("short", "Upbeat & energetic")
    assert plan["bpm"][0] < plan["bpm"][1] and plan["sources"]
