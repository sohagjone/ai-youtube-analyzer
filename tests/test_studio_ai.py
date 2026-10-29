"""AI video understanding, growth/subscriber forecast, competitor comparison and the new Studio flow."""
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src import benchmarks as bm
from src import growth_predictor as gp
from src import shorts_predictor as sp
from src import studio, video_probe as vp
from src import video_understanding as vu
from src.ai_provider import AIProvider
from src.pipeline import run_analysis
from src.sample_data import SampleClient
from src.video_analysis import enrich_videos
from tests.conftest import goto
from tests.test_studio import fake_results, vids
from utils.data_processing import videos_to_df

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
GOOD_JSON = {"topic": "cold brew ratio", "niche": "coffee", "summary": "A creator explains a cold brew ratio.", "key_moments": ["pour", "taste"], "mood": "calm",
             "language": "English", "audience": "home brewers", "title_ideas": ["Cold brew ratio that works", "Cold brew in 5 steps", 7],
             "description": "A quick cold brew guide.", "tags": ["cold brew", "coffee ratio"], "hashtags": ["#coldbrew", "coffee"], "music_mood": "warm lo-fi",
             "rights_flags": ["possible channel logo in the corner"], "confidence": "MEDIUM"}


class FakeAI:
    is_configured = True

    def __init__(self, payload=None, fail=False):
        self.payload, self.fail, self.calls = payload or GOOD_JSON, fail, []

    def complete_json(self, system, user, max_tokens=0, images=None):
        self.calls.append({"system": system, "user": user, "images": images})
        if self.fail:
            raise RuntimeError("model unavailable")
        return self.payload


@pytest.fixture(scope="module")
def sample_results():
    return run_analysis(SampleClient(NOW), "@sample_own", [], sample_size=100, now=NOW, auto_competitors=10)


# ------------------------------------------------------------------ AI provider images
class R:
    status_code = 200

    def __init__(self, body):
        self.b = body

    def json(self):
        return self.b


class PostSess:
    def __init__(self, body):
        self.body, self.last = body, None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last = json
        return R(self.body)


def test_images_are_sent_in_each_providers_format_and_plain_text_otherwise():
    a = AIProvider("anthropic", "k", session=PostSess({"content": [{"type": "text", "text": "ok"}]}))
    a.complete("s", "describe", images=[b"\xff\xd8abc"])
    blocks = a.session.last["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/jpeg" and blocks[-1] == {"type": "text", "text": "describe"}
    o = AIProvider("openai", "k", session=PostSess({"choices": [{"message": {"content": "ok"}}]}))
    o.complete("s", "describe", images=[b"\xff\xd8abc"])
    parts = o.session.last["messages"][1]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and parts[-1]["type"] == "text"
    a.complete("s", "plain")
    assert a.session.last["messages"][0]["content"] == "plain"


# ------------------------------------------------------------------ understanding
def test_understand_validates_and_cleans_the_ai_answer():
    out = vu.understand(FakeAI(), [b"f1", b"f2"], {"duration_seconds": 30, "width": 1080, "height": 1920, "orientation": "vertical"}, "my notes", "hello transcript", "food")
    assert out["topic"] == "cold brew ratio" and out["confidence"] == "medium" and out["hashtags"] == ["coldbrew", "coffee"]
    assert out["title_ideas"] == ["Cold brew ratio that works", "Cold brew in 5 steps", "7"] and out["source"] == "AI: frames + transcript"


def test_understand_sends_frames_and_safety_rules_and_fails_loudly():
    ai = FakeAI()
    vu.understand(ai, [b"1", b"2", b"3"], None, "", None, "")
    call = ai.calls[0]
    assert len(call["images"]) == 3 and "do not identify real people" in call["system"] and "never invent" in call["system"]
    for bad_provider, frames in ((None, [b"x"]), (FakeAI(), []), (FakeAI(payload=["not", "a", "dict"]), [b"x"])):
        with pytest.raises(ValueError):
            vu.understand(bad_provider, frames, None)
    assert vu.understand(FakeAI(payload={"confidence": "certain!"}), [b"x"], None)["confidence"] == "low"


def test_fallback_uses_only_the_users_text():
    f = vu.fallback("My title", "", "notes")
    assert f["topic"] == "My title" and f["confidence"] == "none" and "not analysed" in f["source"] and f["title_ideas"] == [] and f["rights_flags"] == []


def test_frames_are_real_jpegs_and_transcript_is_optional(tmp_path):
    clip = Path("/tmp/clip.mp4")
    exe = vp.ffmpeg_exe()
    import subprocess
    p = tmp_path / "c.mp4"
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=6:s=320x240:r=25", "-f", "lavfi", "-i", "sine=f=440:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    info, frames, transcript = vp.analyze_bytes_full(p.read_bytes(), ".mp4", frames=4, with_transcript=False)
    assert len(frames) == 4 and all(f[:2] == b"\xff\xd8" for f in frames) and transcript is None and info["duration_seconds"] == pytest.approx(6, abs=0.3)
    assert vp.transcribe(p) in (None, "") or isinstance(vp.transcribe(p), str)  # None when faster-whisper is not installed


def test_analyze_bytes_full_deletes_temp_file_even_on_error(monkeypatch):
    import os
    seen = []
    real = vp.analyze_video
    monkeypatch.setattr(vp, "analyze_video", lambda path: (seen.append(path), real(path))[1])
    with pytest.raises(ValueError):
        vp.analyze_bytes_full(b"garbage", ".mp4")
    assert seen and not os.path.exists(seen[0])


# ------------------------------------------------------------------ views for any format + growth forecast
def test_long_form_views_use_28_day_maturity_and_long_form_history():
    rows = vids(20, lambda i: 5000, lambda i: 40 + i * 5, dur=600) + vids(6, lambda i: 99999, lambda i: 10 + i, dur=600, channel="N")  # six videos younger than 28 days
    df = enrich_videos(videos_to_df(rows), NOW)
    df["topic"] = "Espresso"
    res = {"videos": df, "topics": pd.DataFrame({"topic": ["Espresso"], "videos": [26]}), "channel": {"subscribers": 1000}, "competitor_videos": {}, "competitor_channels": {}}
    out = sp.predict_views(res, "Espresso video", 600, None, None, "Long-form")
    assert out["available"] and out["n_shorts"] == 20 and out["baseline"] == 5000 and "long-form" in out["method"]  # the young, viral videos are ignored


def make_forecast_results(subs=10_000, total_views=1_000_000, peers=None):
    r = fake_results(lambda i: 1000, n_short=20)
    r["channel"] = {"subscribers": subs, "total_views": total_views}
    r["competitor_channels"] = peers or {}
    return r


def test_subscriber_math_is_views_times_ratio_with_peer_range():
    peers = {f"P{i}": {"subscribers": s, "total_views": 1_000_000} for i, s in enumerate([5_000, 8_000, 20_000, 30_000])}
    r = make_forecast_results(peers=peers)
    views = {"available": True, "estimate_low": 500.0, "estimate_mid": 1000.0, "estimate_high": 2000.0, "method": "test", "confidence": "High"}
    f = gp.forecast(r, views, "Short", 30)
    t = f["table"].set_index("metric")
    own = 10_000 / 1_000_000
    p25, p75 = np.percentile([0.005, 0.008, 0.02, 0.03], [25, 75])
    assert t.loc["New subscribers", "mid"] == pytest.approx(1000 * own)
    assert t.loc["New subscribers", "low"] == pytest.approx(500 * min(p25, own)) and t.loc["New subscribers", "high"] == pytest.approx(2000 * max(p75, own))
    for m in t.index:
        assert t.loc[m, "low"] <= t.loc[m, "mid"] <= t.loc[m, "high"] and t.loc[m, "low"] >= 0


def test_watch_hours_follow_the_editable_assumption_and_likes_use_own_rates():
    r = make_forecast_results()
    views = {"available": True, "estimate_low": 1000.0, "estimate_mid": 1000.0, "estimate_high": 1000.0, "method": "test", "confidence": "Low"}
    f50 = gp.forecast(r, views, "Long-form", 600, 0.5)
    f25 = gp.forecast(r, views, "Long-form", 600, 0.25)
    h50 = f50["table"].set_index("metric").loc["Watch hours (assumption-based)", "mid"]
    assert h50 == pytest.approx(1000 * 600 * 0.5 / 3600) and f25["table"].set_index("metric").loc["Watch hours (assumption-based)", "mid"] == pytest.approx(h50 / 2)
    likes = f50["table"].set_index("metric").loc["Likes", "mid"]
    assert likes == pytest.approx(1000 * 10 / 1000)  # fixtures have 10 likes on 1,000 views
    assert "NOT measured" in f50["table"].set_index("metric").loc["Watch hours (assumption-based)", "basis"]
    assert gp.default_pct_watched("Short", 30) > gp.default_pct_watched("Long-form", 1500)


def test_forecast_degrades_honestly():
    r = make_forecast_results(subs=None, total_views=None)
    views = {"available": True, "estimate_low": 100.0, "estimate_mid": 200.0, "estimate_high": 400.0, "method": "t", "confidence": "Low"}
    f = gp.forecast(r, views, "Short", 30)
    assert "New subscribers" not in set(f["table"]["metric"])  # no subscriber row without public totals
    assert gp.forecast(r, {"available": False, "reason": "too few"}, "Short", 30) == {"available": False, "reason": "too few"}


# ------------------------------------------------------------------ competitor comparison
def peers_results():
    r = fake_results(lambda i: 1000, n_short=20)
    r["channel"] = {"subscribers": 1000, "total_views": 100_000}
    peers, chans = {}, {}
    for k in range(4):
        rows = vids(20, lambda i, k=k: 2000 + 100 * i, lambda i: 30 + i * 2, dur=40 + k * 5, title_fn=lambda i: f"Peer {i} espresso tips" if i % 2 else f"Grinder review {i}", channel=f"P{k}")
        peers[f"P{k}"] = enrich_videos(videos_to_df(rows), NOW)
        chans[f"P{k}"] = {"subscribers": 10_000, "total_views": 500_000}
    r.update(competitor_videos=peers, competitor_channels=chans)
    return r


def test_comparison_rows_and_verdicts():
    r = peers_results()
    views = sp.predict_views(r, "5 espresso tips", 45, None, None, "Short")
    f = gp.forecast(r, views, "Short", 45)
    t = bm.compare(r, "Short", "5 espresso tips", 45, views, f).set_index("metric")
    assert {"Length (seconds)", "Title length (characters)", "Title words", "Estimated views (size-adjusted peers)", "Expected engagement rate"} <= set(t.index)
    assert t.loc["Length (seconds)", "verdict"] == "Within the typical range"  # peers run 40-55 s
    far = bm.compare(r, "Short", "x" * 120, 170, views, f).set_index("metric")
    assert far.loc["Length (seconds)", "verdict"].startswith("Above") and far.loc["Title length (characters)", "verdict"].startswith("Above")
    # size adjustment: peers get 2,000+ views per 10,000 subs -> about 0.2/sub-unit x your 1,000 subs
    assert t.loc["Estimated views (size-adjusted peers)", "peers_median"] == pytest.approx(0.2 * 1000 * 1.0, rel=0.6)


def test_closest_competitor_videos_rank_by_topic_overlap():
    r = peers_results()
    c = bm.closest_videos(r, "espresso tips for beginners", 5)
    assert len(c) == 5 and c["similarity"].is_monotonic_decreasing and c["title"].str.contains("espresso").all() and c["url"].str.startswith("http").all()
    assert bm.closest_videos(r, "zzzz qqqq").empty and bm.closest_videos({"competitor_videos": {}}, "espresso").empty


# ------------------------------------------------------------------ the Studio flow with AI
def test_run_studio_with_ai_fills_topic_titles_description_tags_and_flags(sample_results):
    ai = FakeAI()
    S = studio.run_studio(sample_results, {"format": "Auto", "tz": "Asia/Dhaka", "music": "licensed", "footage": "all_original"},
                          {"duration_seconds": 40.0, "orientation": "vertical", "width": 1080, "height": 1920, "has_audio": True, "has_video": True, "cuts": {}, "loudness": {}, "silence": {}, "tempo": {}},
                          ai, ["football"], None, datetime(2026, 10, 28, 9, tzinfo=ZoneInfo("Asia/Dhaka")), frames=[b"f1", b"f2"], transcript="spoken words")
    assert S["format"] == "Short" and S["topic_used"] == "cold brew ratio" and S["understanding"]["source"] == "AI: frames + transcript"
    assert "AI" in set(S["titles"]["source"]) and S["titles"]["fit_score"].is_monotonic_decreasing
    assert "A quick cold brew guide." in S["caption"] and "#" in S["caption"] and "cold brew" in S["hashtags"]["hidden_tags"]
    assert any("AI noticed" in i for i in S["copyright"]["findings"]["issue"]) and S["music"]["ai_mood"] == "warm lo-fi"
    assert S["forecast"]["available"] and len(S["comparison"]) > 0 and S["planner"]["plan"].shape[0] >= 1
    assert len(ai.calls) == 1 and len(ai.calls[0]["images"]) == 2
    json.loads(__import__("utils.data_processing", fromlist=["x"]).results_to_json_bytes(S))  # whole result serialises


def test_run_studio_without_ai_never_pretends_to_understand_the_video(sample_results):
    S = studio.run_studio(sample_results, {"format": "Long-form", "title": "My cold brew routine", "topic": "cold brew", "tz": "UTC"}, None, None, None, frames=[b"x"])
    assert S["understanding"]["confidence"] == "none" and S["understanding"]["title_ideas"] == [] and S["ai_note"] == ""
    S2 = studio.run_studio(sample_results, {"format": "Long-form", "topic": "cold brew", "tz": "UTC"}, None, FakeAI(fail=True), None, frames=[b"x"])
    assert "AI understanding failed" in S2["ai_note"] and S2["understanding"]["confidence"] == "none" and len(S2["titles"]) > 0


def test_studio_export_has_new_sheets(sample_results):
    import openpyxl

    from utils.export_studio import build_studio_workbook
    S = studio.run_studio(sample_results, {"format": "Long-form", "topic": "cold brew", "tz": "UTC"}, None, FakeAI(), None, frames=[b"x"])
    names = openpyxl.load_workbook(io.BytesIO(build_studio_workbook(S))).sheetnames
    assert {"AI Understanding", "Predicted Performance", "Competitor Comparison", "Titles", "Upload Plan", "Copyright Risk"} <= set(names)


# ------------------------------------------------------------------ the app
def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()


def test_app_cross_platform_search_and_studio_prediction_sections():
    at = _app()
    [t for t in at.sidebar.text_input if "Your channel" in t.label][0].set_value("@sample_own")
    at.sidebar.button[0].click()
    at.run()
    goto(at, "Trend Radar")
    [b for b in at.button if b.label == "🔍 Find viral videos in my niche"][0].click()
    at.run()
    V = at.session_state["viral"]
    assert set(V["df"]["platform"]) == {"YouTube", "X", "TikTok", "Instagram"} and not at.exception
    assert at.session_state["trend_terms"]  # hashtags found are passed on to the generator
    goto(at, "Video Studio")
    [t for t in at.text_input if t.label.startswith("Topic")][0].set_value("cold brew coffee")
    [b for b in at.button if b.label == "🎬 Analyse my video"][0].click()
    at.run()
    S = at.session_state["studio"]
    assert not at.exception and S["forecast"]["available"] and "comparison" in S and at.radio(key="active_tab").value.endswith("Video Studio")
