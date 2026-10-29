"""Cross-platform viral finder: connectors are tested against mocked HTTP (no network, no real credentials)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import platforms as pf

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
CFG = pf.NICHE_PRESETS["football"]


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body if body is not None else {}

    def json(self):
        return self.body


class Sess:
    """Records every request; replies from a queue (last item repeats)."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


# ------------------------------------------------------------------ niche
def test_niche_presets_and_custom_keywords():
    assert pf.niche_config("football")["label"] == "Football (soccer)" and "premier league" in pf.niche_config("football")["keywords"]
    c = pf.niche_config("football", "premier league goals, messi skills\nlast minute winners")
    assert c["label"] == "Custom" and c["keywords"][0] == "premier league goals" and c["hashtags"][0] == "PremierLeagueGoals"
    assert pf.niche_config("nonsense") == pf.NICHE_PRESETS["football"]


# ------------------------------------------------------------------ X
def test_x_query_is_bounded_and_requires_video():
    q = pf.XConnector.build_query(CFG)
    assert q.startswith("(") and "has:videos" in q and "-is:retweet" in q and '"premier league"' in q
    long_cfg = {"keywords": [f"very long keyword number {i}" for i in range(60)]}
    assert len(pf.XConnector.build_query(long_cfg)) <= 512


def test_x_search_request_and_parsing():
    body = {"data": [{"id": "1", "text": "Amazing goal #football", "created_at": "2026-10-28T06:00:00.000Z", "author_id": "u1",
                      "public_metrics": {"impression_count": 90000, "like_count": 500, "reply_count": 40, "retweet_count": 100, "quote_count": 20},
                      "entities": {"hashtags": [{"tag": "football"}]}}], "includes": {"users": [{"id": "u1", "username": "someone"}]}}
    s = Sess(Resp(200, body))
    posts = pf.XConnector("TOKEN123", session=s).search(CFG, 30, NOW)
    method, url, kw = s.calls[0]
    assert method == "GET" and url.endswith("/2/tweets/search/recent") and kw["headers"]["Authorization"] == "Bearer TOKEN123"
    assert kw["params"]["start_time"] == "2026-10-21T12:00:00Z"  # recent search reaches back 7 days at most, even if 30 were asked
    p = posts[0]
    assert (p["platform"], p["views"], p["likes"], p["comments"], p["shares"], p["url"]) == ("X", 90000, 500, 40, 120, "https://x.com/someone/status/1") and p["hashtags"] == ["football"]


@pytest.mark.parametrize("status,word", [(401, "access denied"), (403, "access denied"), (429, "rate limit"), (500, "failed")])
def test_platform_errors_are_friendly_and_leak_nothing(status, word):
    with pytest.raises(pf.PlatformError) as e:
        pf.XConnector("SECRETTOKEN", session=Sess(Resp(status, {"detail": "SECRETTOKEN"}))).search(CFG, 7, NOW)
    assert word in str(e.value).lower() and "SECRETTOKEN" not in str(e.value)


def test_network_error_is_redacted():
    import requests
    with pytest.raises(pf.PlatformError) as e:
        pf.XConnector("SECRETTOKEN", session=Sess(requests.ConnectionError("failed https://x?token=SECRETTOKEN"))).search(CFG, 7, NOW)
    assert "SECRETTOKEN" not in str(e.value)


# ------------------------------------------------------------------ TikTok Research API
def test_tiktok_token_query_window_and_pagination():
    video = {"id": "7001", "create_time": int((NOW - timedelta(hours=5)).timestamp()), "username": "fan", "video_description": "wow #football",
             "like_count": 100, "comment_count": 10, "share_count": 5, "view_count": 5000, "hashtag_names": ["football"], "video_duration": 30}
    s = Sess(Resp(200, {"access_token": "AT"}), Resp(200, {"data": {"videos": [video], "has_more": True, "cursor": 100}}),
             Resp(200, {"data": {"videos": [dict(video, id="7002")], "has_more": False}}))
    posts = pf.TikTokConnector("KEY", "SEC", session=s).search(CFG, 90, NOW, limit=200)
    tok, q1, q2 = s.calls
    assert tok[1].endswith("/v2/oauth/token/") and tok[2]["data"]["grant_type"] == "client_credentials"
    assert q1[1].endswith("/v2/research/video/query/") and q1[2]["headers"]["Authorization"] == "Bearer AT" and "view_count" in q1[2]["params"]["fields"]
    body = q1[2]["json"]
    assert body["query"]["or"][0]["operation"] == "IN" and body["start_date"] == "20260928" and body["end_date"] == "20261028"  # 30-day maximum window
    assert q2[2]["json"]["cursor"] == 100
    assert [p["id"] for p in posts] == ["7001", "7002"] and posts[0]["views"] == 5000 and posts[0]["url"] == "https://www.tiktok.com/@fan/video/7001"


def test_tiktok_missing_token_is_reported():
    with pytest.raises(pf.PlatformError, match="no access token"):
        pf.TikTokConnector("KEY", "SEC", session=Sess(Resp(200, {}))).search(CFG, 7, NOW)
    assert not pf.TikTokConnector("KEY", "").configured()


# ------------------------------------------------------------------ Instagram hashtag search
def test_instagram_uses_two_hashtags_keeps_only_videos_and_has_no_views():
    media = {"data": [{"id": "m1", "media_type": "VIDEO", "like_count": 900, "comments_count": 30, "permalink": "https://instagram.com/reel/a", "timestamp": "2026-10-28T08:00:00+0000", "caption": "Goal! #football #goals"},
                      {"id": "m2", "media_type": "IMAGE", "like_count": 5000, "comments_count": 1, "permalink": "x", "timestamp": "2026-10-28T08:00:00+0000"}]}
    s = Sess(Resp(200, {"data": [{"id": "H1"}]}), Resp(200, media))
    posts = pf.InstagramConnector("TOK", "IGID", session=s).search(CFG, 7, NOW)
    urls = [c[1] for c in s.calls]
    assert len(s.calls) == 4 and urls[0].endswith("/ig_hashtag_search") and urls[1].endswith("/H1/top_media")  # 2 hashtags x (lookup + media)
    assert [p["id"] for p in posts][:1] == ["m1"] and all(p["views"] is None for p in posts) and posts[0]["hashtags"] == ["football", "goals"]
    assert not any(p["id"] == "m2" for p in posts)


def test_facebook_is_listed_as_unavailable_not_scraped():
    t = pf.status_table(pf.build_connectors(None, {}, sample=False)[1:])
    fb = t[t["platform"] == "Facebook"].iloc[0]
    assert fb["status"] == "unavailable" and "CrowdTangle" in fb["needs"]
    assert set(t[t["status"] == "not connected"]["platform"]) == {"X", "TikTok", "Instagram"}


# ------------------------------------------------------------------ scoring
def post(plat, hours, views, likes=0, comments=0, shares=0):
    return pf._post(plat, f"{plat}{hours}{views}", "t #tag", "u", "a", NOW - timedelta(hours=hours), views, likes, comments, shares)


def test_viral_score_is_a_percentile_within_each_platform():
    posts = [post("X", 10, 1000 * (i + 1), 10 * (i + 1)) for i in range(10)] + [post("TikTok", 10, 10 ** 7 * (i + 1), 1000 * (i + 1)) for i in range(10)]
    df = pf.score_posts(posts, NOW)
    for plat, g in df.groupby("platform"):
        assert g["viral_score"].max() == 100.0 and g["viral_score"].min() == 10.0  # same scale despite 10,000x different raw numbers


def test_engagement_weights_age_floor_and_status():
    df = pf.score_posts([post("X", 0.1, 100, 10, 10, 10)], NOW)
    r = df.iloc[0]
    assert r["engagement"] == 10 * 1 + 10 * 2 + 10 * 3 and r["age_hours"] == 1.0  # age floored at one hour
    old = pf.score_posts([post("X", 100, 1, 0), post("X", 100, 10 ** 6, 1000)], NOW)
    assert old.iloc[0]["status"] == "Viral" and old.iloc[1]["status"] == "Rising"  # not 'Trending now' because older than 48h
    fresh = pf.score_posts([post("X", 5, 1, 0), post("X", 5, 10 ** 6, 1000)], NOW)
    assert fresh.iloc[0]["status"] == "Trending now"


def test_platform_without_views_is_still_scored():
    df = pf.score_posts([pf._post("Instagram", str(i), "t", "u", "a", NOW - timedelta(hours=10), None, 100 * (i + 1), 5, None) for i in range(5)], NOW)
    assert df["view_velocity"].isna().all() and df["viral_score"].notna().all() and df.iloc[0]["likes"] == 500


def test_empty_results_are_handled():
    df = pf.score_posts([], NOW)
    assert df.empty and list(pf.platform_summary(df).columns)[0] == "platform" and pf.what_they_share(df) == []


# ------------------------------------------------------------------ discover
class Boom(pf._Base):
    name, requires = "Boom", "n/a"

    def search(self, *a, **k):
        raise pf.PlatformError("Boom: rate limit reached.")


class Off(pf._Base):
    name, requires = "Off", "OFF_KEY"

    def configured(self):
        return False


def test_discover_isolates_failures_and_skips_unconfigured():
    cons = [Boom(), Off(), pf.SampleConnector("TikTok")]
    df, warn = pf.discover(cons, CFG, 7, now=NOW)
    assert set(df["platform"]) == {"TikTok"} and any("rate limit" in w for w in warn) and any("not connected" in w and "OFF_KEY" in w for w in warn)
    only, _ = pf.discover([pf.SampleConnector("X"), pf.SampleConnector("TikTok")], CFG, 7, selected=["X"], now=NOW)
    assert set(only["platform"]) == {"X"}


def test_sample_connectors_are_labelled_and_deterministic():
    a = pf.SampleConnector("YouTube").search(CFG, 7, NOW)
    b = pf.SampleConnector("YouTube").search(CFG, 7, NOW)
    assert a == b and all(p["title"].startswith("[SAMPLE]") for p in a)
    assert all(p["views"] is None for p in pf.SampleConnector("Instagram").search(CFG, 7, NOW))


def test_top_hashtags_and_what_they_share():
    df, _ = pf.discover(pf.build_connectors(None, {}, sample=True), CFG, 7, now=NOW)
    tags = pf.top_hashtags(df)
    assert len(tags) > 0 and tags["viral_posts"].is_monotonic_decreasing
    lines = pf.what_they_share(df)
    assert any("Descriptive only" in l for l in lines)


# ------------------------------------------------------------------ TikTok error reporting (401 reports)
def test_tiktok_401_on_query_reports_the_step_code_and_the_research_approval_reason():
    err = {"error": {"code": "access_token_invalid", "message": "Access token is invalid", "log_id": "LOG123"}}
    s = Sess(Resp(200, {"access_token": "AT"}), Resp(401, err))
    with pytest.raises(pf.PlatformError) as e:
        pf.TikTokConnector("KEY", "SEC", session=s).search(CFG, 7, NOW)
    msg = str(e.value)
    assert "(video query)" in msg and "access_token_invalid" in msg and "LOG123" in msg and "Research API" in msg and "research.data.basic" in msg
    assert "AT" not in msg.replace("access_token_invalid", "").replace("Access token", "")  # the token itself is never shown


def test_tiktok_bad_client_credentials_are_reported_as_the_token_step():
    with pytest.raises(pf.PlatformError) as e:
        pf.TikTokConnector("KEY", "SEC", session=Sess(Resp(401, {"error": "invalid_client", "error_description": "Client key or secret is incorrect"}))).search(CFG, 7, NOW)
    assert "(token request)" in str(e.value) and "invalid_client" in str(e.value) and "client key/secret" in str(e.value)
    with pytest.raises(pf.PlatformError) as e2:  # TikTok may also answer HTTP 200 with an error body
        pf.TikTokConnector("KEY", "SEC", session=Sess(Resp(200, {"error": "invalid_client", "error_description": "bad secret SEC"}))).search(CFG, 7, NOW)
    assert "token request" in str(e2.value) and "SEC" not in str(e2.value)


def test_error_details_never_leak_credentials_for_any_platform():
    with pytest.raises(pf.PlatformError) as e:
        pf.XConnector("SECRETTOKEN", session=Sess(Resp(403, {"title": "Forbidden", "detail": "token SECRETTOKEN lacks access"}))).search(CFG, 7, NOW)
    assert "SECRETTOKEN" not in str(e.value) and "Forbidden" in str(e.value)
