"""API client tests against a mocked HTTP session (no network, no real key)."""
import pytest

from src.youtube_api import (ChannelNotFoundError, InvalidAPIKeyError, MissingAPIKeyError, QuotaExceededError, YouTubeAPIError,
                             YouTubeClient, estimate_quota)
from utils.data_processing import parse_iso_duration
from utils.validation import ParsedChannel

KEY = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY12345"


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body if body is not None else {}

    def json(self):
        return self._body


class FakeSession:
    """Routes by endpoint; `routes[endpoint]` is a list of responses consumed in order (last one repeats)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        ep = url.rsplit("/", 1)[-1]
        self.calls.append((ep, dict(params)))
        seq = self.routes[ep]
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item


def channel_item(**over):
    base = {"id": "UC" + "a" * 22, "snippet": {"title": "Chan", "publishedAt": "2020-01-01T00:00:00Z"},
            "statistics": {"subscriberCount": "1000", "viewCount": "50000", "videoCount": "20"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UUplaylist"}}}
    base.update(over)
    return base


def video_item(i, **stats):
    return {"id": f"vid{i}", "snippet": {"title": f"Video {i}", "publishedAt": f"2026-0{1 + i % 9}-01T00:00:00Z", "channelId": "UC" + "a" * 22,
                                           "channelTitle": "Chan", "categoryId": "27", "tags": ["x"]},
            "contentDetails": {"duration": "PT1M30S"}, "statistics": stats or {"viewCount": str(100 * i)}}


def client(routes):
    return YouTubeClient(KEY, session=FakeSession(routes), backoff=0)


def test_missing_key():
    with pytest.raises(MissingAPIKeyError):
        YouTubeClient("")


def test_get_channel_by_handle_and_hidden_subscribers():
    c = client({"channels": [Resp(body={"items": [channel_item(statistics={"hiddenSubscriberCount": True, "viewCount": "5", "videoCount": "1"})]})]})
    ch = c.get_channel(ParsedChannel("handle", "@x"))
    assert ch["subscribers"] is None and ch["subscribers_hidden"] is True and ch["total_views"] == 5
    assert c.session.calls[0][1]["forHandle"] == "@x"


def test_channel_not_found():
    c = client({"channels": [Resp(body={"items": []})], "search": [Resp(body={"items": []})]})
    with pytest.raises(ChannelNotFoundError):
        c.get_channel(ParsedChannel("handle", "@nobody"))


def test_quota_exceeded_and_invalid_key_and_no_key_leak():
    q = client({"channels": [Resp(403, {"error": {"message": "quota", "errors": [{"reason": "quotaExceeded"}]}})]})
    with pytest.raises(QuotaExceededError):
        q.get_channel(ParsedChannel("id", "UC" + "a" * 22))
    bad = client({"channels": [Resp(400, {"error": {"message": f"API key not valid {KEY}", "errors": [{"reason": "badRequest"}]}})]})
    with pytest.raises(InvalidAPIKeyError) as e:
        bad.get_channel(ParsedChannel("id", "UC" + "a" * 22))
    assert KEY not in str(e.value)


def test_network_error_message_is_redacted():
    import requests
    c = client({"channels": [requests.ConnectionError(f"failed for url ...key={KEY}")]})
    c.max_retries = 1
    with pytest.raises(YouTubeAPIError) as e:
        c.get_channel(ParsedChannel("id", "UC" + "a" * 22))
    assert KEY not in str(e.value)


def test_pagination_stops_at_requested_size_and_skips_deleted():
    def page(start, n, token):
        items = [{"snippet": {"title": f"V{start + k}"}, "contentDetails": {"videoId": f"vid{start + k}", "videoPublishedAt": "2026-01-01T00:00:00Z"}} for k in range(n)]
        items.insert(0, {"snippet": {"title": "Deleted video"}, "contentDetails": {"videoId": "gone"}})
        body = {"items": items}
        if token:
            body["nextPageToken"] = token
        return Resp(body=body)

    c = client({"playlistItems": [page(0, 50, "T2"), page(50, 50, None)]})
    ids, skipped = c.get_upload_ids("UUplaylist", max_videos=70)
    assert len(ids) == 70 and skipped >= 1 and "gone" not in ids
    assert [call[1].get("pageToken") for call in c.session.calls] == [None, "T2"]


def test_fewer_videos_than_requested_and_unavailable_videos_noted():
    routes = {
        "channels": [Resp(body={"items": [channel_item()]})],
        "playlistItems": [Resp(body={"items": [{"snippet": {"title": f"V{i}"}, "contentDetails": {"videoId": f"vid{i}", "videoPublishedAt": "2026-01-01T00:00:00Z"}} for i in range(1, 4)]})],
        "videos": [Resp(body={"items": [video_item(1), video_item(2)]})],  # vid3 was removed between calls
    }
    b = client(routes).fetch_channel_bundle(ParsedChannel("id", "UC" + "a" * 22), sample_size=10)
    assert len(b["videos"]) == 2
    assert any("unavailable" in n for n in b["notes"]) and any("Only 2" in n for n in b["notes"])


def test_missing_likes_comments_are_none_not_zero():
    v = YouTubeClient._normalise_video(video_item(1, viewCount="10"))
    assert v["likes"] is None and v["comments"] is None and v["views"] == 10


def test_batching_of_video_ids():
    c = client({"videos": [Resp(body={"items": []})]})
    c.get_videos([f"v{i}" for i in range(120)])
    assert [len(call[1]["id"].split(",")) for call in c.session.calls] == [50, 50, 20]


def test_empty_channel_without_uploads_playlist():
    item = channel_item(contentDetails={})
    b = client({"channels": [Resp(body={"items": [item]})]}).fetch_channel_bundle(ParsedChannel("id", item["id"]))
    assert b["videos"] == [] and b["notes"]


def test_duration_parsing_and_quota_estimate():
    assert parse_iso_duration("PT1H2M3S") == 3723 and parse_iso_duration("PT45S") == 45 and parse_iso_duration("bogus") == 0
    assert estimate_quota(50, 1) == 3 and estimate_quota(100, 2) == 10
