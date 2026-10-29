"""Cross-platform viral/trending finder by niche, using OFFICIAL APIs only (no scraping).

Connectors (each works only when the app owner has configured its credentials in secrets):
  YouTube    - YouTube Data API v3 key                       (search.list + videos.list)
  X          - X API v2 bearer token (plan must allow search) (GET /2/tweets/search/recent, has:videos)
  TikTok     - TikTok Research API client key/secret          (POST /v2/research/video/query/) - approved researchers only
  Instagram  - Meta Graph API token + IG business account id  (ig_hashtag_search -> top_media; needs Instagram Public Content Access)
Facebook has no public keyword-search API for videos (CrowdTangle was shut down; Meta Content Library is for approved
researchers), so it is listed as unavailable rather than scraped.

Metrics differ by platform (Instagram returns no view counts), so 'viral score' is a percentile WITHIN each platform's results.
It means 'moving unusually fast right now compared with the other results found on that platform', not an absolute comparison.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

NICHE_PRESETS = {
    "football": {"label": "Football (soccer)", "keywords": ["football", "soccer", "premier league", "champions league", "la liga", "world cup", "goal"],
                 "hashtags": ["football", "soccer", "premierleague", "championsleague", "goals"]},
    "sports": {"label": "Sports (general)", "keywords": ["sports", "highlights", "basketball", "cricket", "tennis", "boxing"], "hashtags": ["sports", "highlights", "basketball", "cricket"]},
    "gaming": {"label": "Gaming", "keywords": ["gaming", "gameplay", "minecraft", "fortnite", "valorant", "roblox"], "hashtags": ["gaming", "gameplay", "minecraft", "fortnite"]},
    "tech": {"label": "Tech", "keywords": ["tech", "smartphone", "iphone", "android", "ai tools", "gadgets"], "hashtags": ["tech", "gadgets", "ai", "iphone"]},
    "education": {"label": "Education / how-to", "keywords": ["tutorial", "how to", "explained", "learn", "study tips"], "hashtags": ["education", "learn", "tutorial", "studytips"]},
    "finance": {"label": "Finance / business", "keywords": ["investing", "personal finance", "stock market", "crypto", "side hustle"], "hashtags": ["finance", "investing", "money", "crypto"]},
    "food": {"label": "Food / cooking", "keywords": ["recipe", "cooking", "street food", "baking", "meal prep"], "hashtags": ["recipe", "cooking", "food", "streetfood"]},
    "travel": {"label": "Travel", "keywords": ["travel", "travel vlog", "hidden gems", "city tour"], "hashtags": ["travel", "travelvlog", "wanderlust"]},
    "fitness": {"label": "Fitness", "keywords": ["workout", "fitness", "gym", "home workout"], "hashtags": ["fitness", "workout", "gym", "homeworkout"]},
    "music": {"label": "Music", "keywords": ["new music", "cover song", "guitar", "producer"], "hashtags": ["music", "cover", "guitar", "newmusic"]},
    "beauty": {"label": "Beauty / fashion", "keywords": ["makeup", "skincare", "fashion", "hair"], "hashtags": ["makeup", "skincare", "fashion", "grwm"]},
    "comedy": {"label": "Comedy", "keywords": ["funny", "comedy", "skit", "pranks"], "hashtags": ["funny", "comedy", "skit", "pranks"]},
}
UNSUPPORTED = {"Facebook": "No public keyword-search API for videos (CrowdTangle was shut down; Meta Content Library is for approved researchers). Not scraped."}
ENGAGEMENT_WEIGHTS = {"likes": 1.0, "comments": 2.0, "shares": 3.0}


class PlatformError(Exception):
    """Platform problem with a user-safe message (never contains credentials)."""


def niche_config(niche: str, custom_keywords: str = "") -> dict:
    custom = [k.strip() for k in re.split(r"[,\n]", custom_keywords or "") if k.strip()]
    if custom:
        tags = [re.sub(r"[^0-9A-Za-z]", "", k.title()) for k in custom]
        return {"label": "Custom", "keywords": custom[:8], "hashtags": [t for t in tags if t][:6]}
    return NICHE_PRESETS.get(niche, NICHE_PRESETS["football"])


def _post(platform: str, pid: str, title: str, url: str, author: str, published: datetime, views=None, likes=None, comments=None,
          shares=None, duration=None, hashtags=None, media="video") -> dict:
    return {"platform": platform, "id": str(pid), "title": (title or "").strip().replace("\n", " ")[:200], "url": url, "author": author or "",
            "published_at": published, "views": views, "likes": likes, "comments": comments, "shares": shares, "duration_seconds": duration,
            "hashtags": list(hashtags or []), "media": media}


class _Base:
    name = "?"
    requires = ""

    def __init__(self, session: requests.Session | None = None, timeout: float = 20.0):
        self.session = session or requests.Session()
        self.timeout = timeout
        self._secrets: list[str] = []

    def configured(self) -> bool:
        return True

    def _safe(self, text: str) -> str:
        for sct in self._secrets:
            if sct:
                text = text.replace(sct, "***")
        return text

    def _detail(self, r) -> str:
        """The platform's own error code/message (safe to show; secrets are redacted)."""
        try:
            body = r.json()
        except ValueError:
            return ""
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            parts = [str(err.get("code", "")), str(err.get("message", "")), f"log_id {err['log_id']}" if err.get("log_id") else ""]
        elif isinstance(err, str):
            parts = [err, str(body.get("error_description", ""))]
        else:
            parts = [str(body.get("title", "")) if isinstance(body, dict) else "", str(body.get("detail", "")) if isinstance(body, dict) else ""]
        text = " - ".join(p for p in parts if p and p != "ok")
        return self._safe(text)[:240]

    def _hint(self, step: str, detail: str) -> str:
        return "Check the credentials and that your plan/approval covers this endpoint."

    def _request(self, method: str, url: str, step: str = "", **kw):
        try:
            r = self.session.request(method, url, timeout=self.timeout, **kw)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise PlatformError(self._safe(f"{self.name}: network problem ({type(exc).__name__}).")) from None
        if r.status_code in (401, 403):
            detail = self._detail(r)
            raise PlatformError(f"{self.name}{' (' + step + ')' if step else ''}: access denied (HTTP {r.status_code}"
                                f"{', ' + detail if detail else ''}). {self._hint(step, detail)}")
        if r.status_code == 429:
            raise PlatformError(f"{self.name}: rate limit reached. Try again later.")
        if r.status_code >= 400:
            detail = self._detail(r)
            raise PlatformError(self._safe(f"{self.name}{' (' + step + ')' if step else ''}: request failed (HTTP {r.status_code}{', ' + detail if detail else ''})."))
        try:
            return r.json()
        except ValueError:
            raise PlatformError(f"{self.name}: unreadable response.") from None


class YouTubeConnector(_Base):
    name, requires = "YouTube", "YOUTUBE_API_KEY"

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self.client = client

    def search(self, cfg: dict, days: int, now: datetime, limit: int = 25) -> list[dict]:
        from src.youtube_api import YouTubeAPIError
        out: dict[str, dict] = {}
        try:
            for q in [f"{k}" for k in cfg["keywords"][:2]]:
                for v in self.client.search_recent_videos(q, now - timedelta(days=days), limit, "viewCount"):
                    pub = pd.to_datetime(v["published_at"], utc=True).to_pydatetime()
                    out[v["video_id"]] = _post("YouTube", v["video_id"], v["title"], v["url"], v.get("channel_title"), pub, v["views"], v.get("likes"),
                                               v.get("comments"), None, v.get("duration_seconds"), [t for t in (v.get("tags") or [])][:10])
        except YouTubeAPIError as exc:
            raise PlatformError(f"YouTube: {exc}") from None
        return list(out.values())


class XConnector(_Base):
    name, requires = "X", "X_BEARER_TOKEN (a plan that allows search; has:videos needs Basic or higher)"
    URL = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, bearer: str | None, **kw):
        super().__init__(**kw)
        self.bearer = (bearer or "").strip()
        self._secrets = [self.bearer]

    def configured(self) -> bool:
        return bool(self.bearer)

    @staticmethod
    def build_query(cfg: dict, lang: str | None = None) -> str:
        terms = [f'"{k}"' if " " in k else k for k in cfg["keywords"]]
        q = "(" + " OR ".join(terms) + ") has:videos -is:retweet" + (f" lang:{lang}" if lang else "")
        while len(q) > 500 and len(terms) > 1:  # recent search queries are limited to 512 characters
            terms.pop()
            q = "(" + " OR ".join(terms) + ") has:videos -is:retweet" + (f" lang:{lang}" if lang else "")
        return q

    def search(self, cfg: dict, days: int, now: datetime, limit: int = 100) -> list[dict]:
        params = {"query": self.build_query(cfg), "max_results": max(10, min(limit, 100)), "sort_order": "relevancy",
                  "tweet.fields": "created_at,public_metrics,entities,lang", "expansions": "author_id", "user.fields": "username",
                  "start_time": (now - timedelta(days=min(days, 7))).strftime("%Y-%m-%dT%H:%M:%SZ")}
        data = self._request("GET", self.URL, params=params, headers={"Authorization": f"Bearer {self.bearer}"})
        users = {u["id"]: u.get("username", "") for u in (data.get("includes") or {}).get("users", [])}
        out = []
        for t in data.get("data", []) or []:
            m = t.get("public_metrics") or {}
            uname = users.get(t.get("author_id"), "i")
            out.append(_post("X", t["id"], t.get("text", ""), f"https://x.com/{uname}/status/{t['id']}", uname,
                             pd.to_datetime(t["created_at"], utc=True).to_pydatetime(), m.get("impression_count"), m.get("like_count"), m.get("reply_count"),
                             (m.get("retweet_count") or 0) + (m.get("quote_count") or 0), None,
                             [h["tag"] for h in (t.get("entities") or {}).get("hashtags", [])]))
        return out


class TikTokConnector(_Base):
    name, requires = "TikTok", "TIKTOK_CLIENT_KEY + TIKTOK_CLIENT_SECRET (TikTok Research API - approved researchers only)"
    TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
    QUERY_URL = "https://open.tiktokapis.com/v2/research/video/query/"
    FIELDS = "id,create_time,username,region_code,video_description,like_count,comment_count,share_count,view_count,hashtag_names,video_duration"

    def __init__(self, client_key: str | None, client_secret: str | None, **kw):
        super().__init__(**kw)
        self.key, self.secret = (client_key or "").strip(), (client_secret or "").strip()
        self._secrets = [self.key, self.secret]

    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def _hint(self, step: str, detail: str) -> str:
        if step == "token request":
            return "TikTok rejected the client key/secret. Copy them again from the app in the TikTok developer portal."
        return ("This usually means the app is not approved for the Research API (scope research.data.basic). TikTok grants that only to approved research "
                "projects - a normal developer account, or a creator/commercial account, cannot query videos. See the TikTok Research API eligibility page.")

    def _token(self) -> str:
        data = self._request("POST", self.TOKEN_URL, step="token request", data={"client_key": self.key, "client_secret": self.secret, "grant_type": "client_credentials"},
                             headers={"Content-Type": "application/x-www-form-urlencoded", "Cache-Control": "no-cache"})
        tok = data.get("access_token")
        if not tok:
            why = self._safe(str(data.get("error_description") or data.get("error") or ""))[:200]
            raise PlatformError("TikTok (token request): no access token returned" + (f" ({why})" if why else "") + ". Check the client key/secret.")
        self._secrets.append(tok)
        return tok

    def search(self, cfg: dict, days: int, now: datetime, limit: int = 100) -> list[dict]:
        token = self._token()
        days = min(days, 30)  # the Research API allows at most a 30-day window per query
        body = {"query": {"or": [{"operation": "IN", "field_name": "hashtag_name", "field_values": cfg["hashtags"][:10]},
                                 {"operation": "IN", "field_name": "keyword", "field_values": cfg["keywords"][:10]}]},
                "max_count": 100, "cursor": 0, "start_date": (now - timedelta(days=days)).strftime("%Y%m%d"), "end_date": now.strftime("%Y%m%d")}
        out: list[dict] = []
        for _ in range(max(1, limit // 100)):
            data = self._request("POST", self.QUERY_URL, step="video query", params={"fields": self.FIELDS}, json=body,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            payload = data.get("data") or {}
            for v in payload.get("videos", []) or []:
                out.append(_post("TikTok", v["id"], v.get("video_description", ""), f"https://www.tiktok.com/@{v.get('username', '')}/video/{v['id']}",
                                 v.get("username"), datetime.fromtimestamp(int(v["create_time"]), tz=timezone.utc), v.get("view_count"), v.get("like_count"),
                                 v.get("comment_count"), v.get("share_count"), v.get("video_duration"), v.get("hashtag_names")))
            if not payload.get("has_more"):
                break
            body["cursor"] = payload.get("cursor", 0)
        return out


class InstagramConnector(_Base):
    name, requires = "Instagram", "META_ACCESS_TOKEN + INSTAGRAM_BUSINESS_ACCOUNT_ID (Instagram Public Content Access feature)"
    BASE = "https://graph.facebook.com/v21.0"

    def __init__(self, token: str | None, ig_user_id: str | None, **kw):
        super().__init__(**kw)
        self.token, self.user_id = (token or "").strip(), (ig_user_id or "").strip()
        self._secrets = [self.token]

    def configured(self) -> bool:
        return bool(self.token and self.user_id)

    def search(self, cfg: dict, days: int, now: datetime, limit: int = 50) -> list[dict]:
        out: list[dict] = []
        for tag in cfg["hashtags"][:2]:  # Meta allows only 30 unique hashtags per 7 days, so use as few as possible
            found = self._request("GET", f"{self.BASE}/ig_hashtag_search", params={"user_id": self.user_id, "q": tag, "access_token": self.token})
            ids = [d["id"] for d in found.get("data", [])]
            if not ids:
                continue
            media = self._request("GET", f"{self.BASE}/{ids[0]}/top_media", params={
                "user_id": self.user_id, "fields": "id,media_type,comments_count,like_count,permalink,timestamp,caption", "limit": min(limit, 50), "access_token": self.token})
            for m in media.get("data", []):
                if m.get("media_type") != "VIDEO":
                    continue
                cap = m.get("caption", "")
                out.append(_post("Instagram", m["id"], cap, m.get("permalink", ""), "", pd.to_datetime(m["timestamp"], utc=True).to_pydatetime(), None,
                                 m.get("like_count"), m.get("comments_count"), None, None, re.findall(r"#(\w+)", cap)[:12], "reel/video"))
        return out


class SampleConnector(_Base):
    """Illustrative data for demo mode. Everything is labelled [SAMPLE]."""

    def __init__(self, platform: str, **kw):
        super().__init__(**kw)
        self.name, self.requires = platform, "demo"

    def search(self, cfg: dict, days: int, now: datetime, limit: int = 25) -> list[dict]:
        rng = random.Random(int(hashlib.sha256((self.name + cfg["label"]).encode()).hexdigest()[:8], 16))
        out = []
        for i in range(min(limit, 25)):
            age_h = rng.uniform(2, days * 24)
            views = int(rng.lognormvariate(10, 1.3))
            kw = rng.choice(cfg["keywords"])
            out.append(_post(self.name, f"{self.name}{i}", f"[SAMPLE] {kw.title()} moment you have to see #{rng.choice(cfg['hashtags'])}", f"https://example.com/{self.name}/{i}",
                             f"creator{rng.randint(1, 40)}", now - timedelta(hours=age_h), None if self.name == "Instagram" else views, int(views * rng.uniform(0.02, 0.09)),
                             int(views * rng.uniform(0.001, 0.01)), int(views * rng.uniform(0.0005, 0.01)) if self.name != "YouTube" else None,
                             rng.choice([15, 28, 45, 60, 200, 600]), rng.sample(cfg["hashtags"], k=min(2, len(cfg["hashtags"])))))
        return out


def build_connectors(youtube_client, secrets: dict, sample: bool = False) -> list:
    if sample:
        return [SampleConnector(n) for n in ("YouTube", "X", "TikTok", "Instagram")]
    return [YouTubeConnector(youtube_client), XConnector(secrets.get("X_BEARER_TOKEN")),
            TikTokConnector(secrets.get("TIKTOK_CLIENT_KEY"), secrets.get("TIKTOK_CLIENT_SECRET")),
            InstagramConnector(secrets.get("META_ACCESS_TOKEN"), secrets.get("INSTAGRAM_BUSINESS_ACCOUNT_ID"))]


def status_table(connectors: list) -> pd.DataFrame:
    rows = [{"platform": c.name, "status": "connected" if c.configured() else "not connected", "needs": c.requires} for c in connectors]
    rows += [{"platform": n, "status": "unavailable", "needs": note} for n, note in UNSUPPORTED.items()]
    return pd.DataFrame(rows)


def score_posts(posts: list[dict], now: datetime) -> pd.DataFrame:
    """Normalise to a DataFrame and add velocity and per-platform viral score (percentile, 0-100)."""
    cols = ["platform", "id", "title", "url", "author", "published_at", "views", "likes", "comments", "shares", "duration_seconds", "hashtags", "media"]
    df = pd.DataFrame(posts, columns=cols)
    if df.empty:
        return df.assign(age_hours=[], engagement=[], engagement_velocity=[], view_velocity=[], viral_score=[], status=[])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    for c in ("views", "likes", "comments", "shares"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["age_hours"] = ((pd.Timestamp(now) - df["published_at"]).dt.total_seconds() / 3600).clip(lower=1.0)
    df["engagement"] = sum(df[c].fillna(0) * w for c, w in ENGAGEMENT_WEIGHTS.items())
    df["engagement_velocity"] = df["engagement"] / df["age_hours"]
    df["view_velocity"] = df["views"] / df["age_hours"]
    score = pd.Series(np.nan, index=df.index)
    for plat, g in df.groupby("platform"):
        ev = g["engagement_velocity"].rank(pct=True)
        vv = g["view_velocity"].rank(pct=True) if g["view_velocity"].notna().any() else None
        score.loc[g.index] = 100 * (0.5 * ev + 0.5 * vv.fillna(ev) if vv is not None else ev)
    df["viral_score"] = score.round(1)
    df["status"] = np.where((df["age_hours"] <= 48) & (df["viral_score"] >= 80), "Trending now", np.where(df["viral_score"] >= 80, "Viral", "Rising"))
    df["engagement_rate"] = np.where(df["views"] > 0, df["engagement"] / df["views"], np.nan)
    return df.sort_values("viral_score", ascending=False).reset_index(drop=True)


def discover(connectors: list, niche: dict, days: int = 7, selected: list[str] | None = None, now: datetime | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Run every selected, configured connector. A failing platform becomes a warning; the others still return."""
    now = now or datetime.now(timezone.utc)
    posts: list[dict] = []
    warnings: list[str] = []
    for c in connectors:
        if selected is not None and c.name not in selected:
            continue
        if not c.configured():
            warnings.append(f"{c.name}: not connected ({c.requires}).")
            continue
        try:
            got = c.search(niche, days, now)
            if not got:
                warnings.append(f"{c.name}: no matching videos found for this niche/time window.")
            posts += got
        except PlatformError as exc:
            warnings.append(str(exc))
    return score_posts(posts, now), warnings


def top_hashtags(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    top = df[df["viral_score"] >= 60] if len(df) else df
    c: Counter[str] = Counter()
    for _, r in top.iterrows():
        for h in set(h.lower().lstrip("#") for h in (r["hashtags"] or [])):
            c[h] += 1
        for h in re.findall(r"#(\w+)", r["title"]):
            c[h.lower()] += 1
    return pd.DataFrame(c.most_common(n), columns=["hashtag", "viral_posts"])


def platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["platform", "videos", "trending_now", "median_age_hours", "median_duration_s"])
    return df.groupby("platform").agg(videos=("id", "count"), trending_now=("status", lambda s: int((s == "Trending now").sum())),
                                      median_age_hours=("age_hours", "median"), median_duration_s=("duration_seconds", "median")).reset_index()


def what_they_share(df: pd.DataFrame, top_n: int = 15) -> list[str]:
    if df.empty:
        return []
    top = df.head(top_n)
    out = []
    if top["duration_seconds"].notna().any():
        short = float((top["duration_seconds"].dropna() <= 60).mean())
        out.append(f"{short:.0%} of the fastest videos are 60 seconds or shorter (median {top['duration_seconds'].median():.0f}s).")
    tags = top_hashtags(df, 5)
    if len(tags):
        out.append("Most common hashtags among the high scorers: " + ", ".join("#" + h for h in tags["hashtag"]) + ".")
    out.append("Descriptive only: this shows what is moving now, not what causes views. Do not copy other creators' videos, titles or thumbnails.")
    return out
