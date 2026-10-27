"""Thin, defensive client for the official YouTube Data API v3 (public data only)."""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import requests

from utils.data_processing import parse_iso_duration
from utils.validation import ParsedChannel

BASE_URL = "https://www.googleapis.com/youtube/v3"
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"}
PLACEHOLDER_TITLES = {"deleted video", "private video"}

CATEGORY_NAMES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals",
    "17": "Sports", "19": "Travel & Events", "20": "Gaming", "22": "People & Blogs",
    "23": "Comedy", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
}


class YouTubeAPIError(Exception):
    """Base class for all API problems (messages are safe to show to users)."""


class MissingAPIKeyError(YouTubeAPIError):
    pass


class InvalidAPIKeyError(YouTubeAPIError):
    pass


class QuotaExceededError(YouTubeAPIError):
    pass


class ChannelNotFoundError(YouTubeAPIError):
    pass


def estimate_quota(sample_size: int, n_channels: int = 1, custom_url_lookups: int = 0) -> int:
    """Rough quota units: channels.list 1 + playlistItems 1/page + videos.list 1/50 videos; search = 100."""
    pages = max(1, math.ceil(sample_size / 50))
    return n_channels * (1 + pages + pages) + custom_url_lookups * 100


class YouTubeClient:
    def __init__(self, api_key: str | None, session: requests.Session | None = None,
                 timeout: float = 15.0, max_retries: int = 2, backoff: float = 0.8):
        if not api_key or not str(api_key).strip():
            raise MissingAPIKeyError("No YouTube API key configured. Add YOUTUBE_API_KEY to secrets or the environment.")
        self._api_key = str(api_key).strip()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.units_estimate = 0  # informational only

    # ------------------------------------------------------------------ low level
    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict:
        query = {**params, "key": self._api_key}
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(f"{BASE_URL}/{endpoint}", params=query, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = self._redact(f"Network problem talking to YouTube ({type(exc).__name__}).")
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise YouTubeAPIError(last_error) from None
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    raise YouTubeAPIError("YouTube returned an unreadable response.") from None
            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.backoff * (attempt + 1))
                continue
            self._raise_for_error(resp)
        raise YouTubeAPIError(last_error)

    def _raise_for_error(self, resp: requests.Response) -> None:
        reason, message = "", ""
        try:
            err = resp.json().get("error", {})
            message = str(err.get("message", ""))
            errors = err.get("errors") or [{}]
            reason = str(errors[0].get("reason", ""))
            if not reason and err.get("details"):
                reason = str(err["details"][0].get("reason", ""))
        except (ValueError, AttributeError, IndexError, KeyError, TypeError):
            pass
        message = self._redact(message)
        if reason in QUOTA_REASONS:
            raise QuotaExceededError("YouTube API quota exhausted. Quota resets daily (Pacific Time); try later or use sample-data mode.")
        if reason in {"keyInvalid", "API_KEY_INVALID"} or "API key not valid" in message:
            raise InvalidAPIKeyError("The YouTube API key was rejected. Check that it is correct and that YouTube Data API v3 is enabled.")
        if reason in {"accessNotConfigured", "forbidden", "ipRefererBlocked", "API_KEY_SERVICE_BLOCKED"}:
            raise YouTubeAPIError(f"Access denied by YouTube ({reason}). {message}".strip())
        if resp.status_code == 404:
            raise ChannelNotFoundError("YouTube could not find that resource.")
        raise YouTubeAPIError(f"YouTube API error {resp.status_code}: {message or reason or 'unknown'}")

    # ------------------------------------------------------------------ channels
    def get_channel(self, ref: ParsedChannel) -> dict:
        params: dict[str, Any] = {"part": "snippet,statistics,contentDetails"}
        if ref.kind == "id":
            params["id"] = ref.value
        elif ref.kind == "handle":
            params["forHandle"] = ref.value
        elif ref.kind == "username":
            params["forUsername"] = ref.value
        elif ref.kind == "custom":
            params["id"] = self._search_channel_id(ref.value)
        else:
            raise ChannelNotFoundError(f"Unsupported channel reference type: {ref.kind}")
        data = self._get("channels", params)
        items = data.get("items") or []
        if not items and ref.kind == "handle":  # bare names are parsed as handles; try a name search before giving up
            params = {"part": "snippet,statistics,contentDetails", "id": self._search_channel_id(ref.value.lstrip("@"))}
            items = (self._get("channels", params).get("items")) or []
        if not items:
            raise ChannelNotFoundError(f"No channel found for '{ref.value}'.")
        return self._normalise_channel(items[0])

    def _search_channel_id(self, name: str) -> str:
        data = self._get("search", {"part": "snippet", "type": "channel", "q": name, "maxResults": 1})
        items = data.get("items") or []
        if not items:
            raise ChannelNotFoundError(f"No channel found for '{name}'.")
        cid = (items[0].get("id") or {}).get("channelId") or (items[0].get("snippet") or {}).get("channelId")
        if not cid:
            raise ChannelNotFoundError(f"No channel found for '{name}'.")
        return cid

    @staticmethod
    def _normalise_channel(item: dict) -> dict:
        sn, st = item.get("snippet", {}) or {}, item.get("statistics", {}) or {}
        cd = item.get("contentDetails", {}) or {}
        hidden = bool(st.get("hiddenSubscriberCount"))

        def _int(key: str) -> int | None:
            try:
                return int(st[key])
            except (KeyError, TypeError, ValueError):
                return None

        thumbs = sn.get("thumbnails", {}) or {}
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
        cid = item.get("id", "")
        return {
            "channel_id": cid,
            "title": sn.get("title", ""),
            "description": sn.get("description", ""),
            "custom_url": sn.get("customUrl"),
            "created_at": sn.get("publishedAt"),
            "country": sn.get("country"),
            "subscribers": None if hidden else _int("subscriberCount"),
            "subscribers_hidden": hidden,
            "total_views": _int("viewCount"),
            "total_videos": _int("videoCount"),
            "uploads_playlist": (cd.get("relatedPlaylists") or {}).get("uploads"),
            "thumbnail_url": thumb,
            "url": f"https://www.youtube.com/channel/{cid}" if cid else None,
        }

    # ------------------------------------------------------------------ videos
    def get_upload_ids(self, playlist_id: str, max_videos: int = 50,
                       date_from: datetime | None = None, date_to: datetime | None = None) -> tuple[list[str], int]:
        """Page through the uploads playlist (newest first). Returns (video_ids, skipped_placeholders)."""
        ids: list[str] = []
        skipped = 0
        token: str | None = None
        while len(ids) < max_videos:
            params: dict[str, Any] = {"part": "snippet,contentDetails", "playlistId": playlist_id, "maxResults": 50}
            if token:
                params["pageToken"] = token
            data = self._get("playlistItems", params)
            stop = False
            for it in data.get("items", []):
                title = ((it.get("snippet") or {}).get("title") or "").strip().lower()
                vid = (it.get("contentDetails") or {}).get("videoId")
                if not vid:
                    skipped += 1
                    continue
                if title in PLACEHOLDER_TITLES:
                    skipped += 1
                    continue
                published = _parse_dt((it.get("contentDetails") or {}).get("videoPublishedAt")
                                      or (it.get("snippet") or {}).get("publishedAt"))
                if published and date_to and published > date_to:
                    continue
                if published and date_from and published < date_from:
                    stop = True  # playlist is newest-first, so everything after this is older
                    break
                ids.append(vid)
                if len(ids) >= max_videos:
                    break
            token = data.get("nextPageToken")
            if stop or not token:
                break
        return ids, skipped

    def get_videos(self, video_ids: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            data = self._get("videos", {"part": "snippet,contentDetails,statistics", "id": ",".join(batch), "maxResults": 50})
            out.extend(self._normalise_video(v) for v in data.get("items", []))
        return out

    @staticmethod
    def _normalise_video(item: dict) -> dict:
        sn, st = item.get("snippet", {}) or {}, item.get("statistics", {}) or {}
        cd = item.get("contentDetails", {}) or {}

        def _opt(key: str) -> int | None:
            try:
                return int(st[key])
            except (KeyError, TypeError, ValueError):
                return None

        thumbs = sn.get("thumbnails", {}) or {}
        thumb = next((thumbs[k]["url"] for k in ("maxres", "standard", "high", "medium", "default")
                      if k in thumbs and thumbs[k].get("url")), None)
        vid = item.get("id", "")
        return {
            "video_id": vid,
            "title": sn.get("title", ""),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "published_at": sn.get("publishedAt"),
            "duration_seconds": parse_iso_duration(cd.get("duration")),
            "views": _opt("viewCount") or 0,
            "likes": _opt("likeCount"),
            "comments": _opt("commentCount"),
            "description": sn.get("description", ""),
            "tags": sn.get("tags") or [],
            "thumbnail_url": thumb,
            "category": CATEGORY_NAMES.get(str(sn.get("categoryId", "")), None),
            "channel_id": sn.get("channelId"),
            "channel_title": sn.get("channelTitle"),
        }

    def fetch_channel_bundle(self, ref: ParsedChannel, sample_size: int = 50,
                             date_from: datetime | None = None, date_to: datetime | None = None) -> dict:
        """Channel info plus up to sample_size public videos. Returns {channel, videos, notes}."""
        sample_size = max(1, min(int(sample_size), 500))
        channel = self.get_channel(ref)
        notes: list[str] = []
        playlist = channel.get("uploads_playlist")
        if not playlist:
            return {"channel": channel, "videos": [], "notes": ["This channel exposes no uploads playlist."]}
        ids, skipped = self.get_upload_ids(playlist, sample_size, date_from, date_to)
        videos = self.get_videos(ids)
        unavailable = len(ids) - len(videos)
        if skipped or unavailable > 0:
            notes.append(f"{skipped + max(unavailable, 0)} deleted, private or unavailable video(s) were skipped.")
        if len(videos) < sample_size:
            notes.append(f"Only {len(videos)} public video(s) available for the selected sample/date range (requested {sample_size}).")
        return {"channel": channel, "videos": videos, "notes": notes}

    # ------------------------------------------------------------------ competitor discovery
    def find_channel_candidates(self, queries: list[str], max_per_query: int = 50,
                                published_after: datetime | None = None) -> list[dict]:
        """Channels that publish recent videos matching the queries (100 quota units per query).
        Returns [{channel_id, title, hits}] sorted by how many query results each channel appeared in."""
        found: dict[str, dict] = {}
        for q in queries:
            if not q or not q.strip():
                continue
            params: dict[str, Any] = {"part": "snippet", "type": "video", "q": q.strip(),
                                      "maxResults": max(1, min(int(max_per_query), 50)), "order": "relevance"}
            if published_after:
                params["publishedAfter"] = published_after.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            data = self._get("search", params)
            seen: set[str] = set()
            for item in data.get("items", []):
                sn = item.get("snippet") or {}
                cid = sn.get("channelId")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                entry = found.setdefault(cid, {"channel_id": cid, "title": sn.get("channelTitle", ""), "hits": 0})
                entry["hits"] += 1
        return sorted(found.values(), key=lambda e: -e["hits"])

    def get_channels_by_ids(self, channel_ids: list[str]) -> list[dict]:
        """Batch channel lookup (1 quota unit per 50 channels)."""
        out: list[dict] = []
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i + 50]
            data = self._get("channels", {"part": "snippet,statistics,contentDetails", "id": ",".join(batch), "maxResults": 50})
            out.extend(self._normalise_channel(it) for it in data.get("items", []))
        return out


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
