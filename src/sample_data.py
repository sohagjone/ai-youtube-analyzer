"""ILLUSTRATIVE sample data generator. Nothing here is real YouTube data; every name is prefixed [SAMPLE]."""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone

from utils.validation import ParsedChannel

TOPIC_POOLS = {
    "own": ["espresso", "pour over", "coffee grinder", "home roasting"],
    "rival_a": ["espresso", "latte art", "milk steaming", "cold brew"],
    "rival_b": ["cold brew", "coffee beans", "pour over", "brew ratios"],
    "default": ["espresso", "cold brew", "latte art", "coffee beans"],
}
TEMPLATES = ["How to make {t} at home", "{t} mistakes beginners make", "5 tips for better {t}", "{t} explained simply",
             "Is {t} worth it?", "{t} vs the alternatives: honest test", "My {t} routine", "{t} gear guide"]


class SampleClient:
    """Drop-in replacement for YouTubeClient that returns deterministic illustrative data."""

    is_sample = True

    def __init__(self, now: datetime | None = None):
        self.now = now or datetime.now(timezone.utc)

    def fetch_channel_bundle(self, ref: ParsedChannel, sample_size: int = 50,
                             date_from: datetime | None = None, date_to: datetime | None = None) -> dict:
        seed = int(hashlib.sha256(ref.value.lower().encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        pool_key = "own" if rng.random() < 0 else None
        value = ref.value.lower()
        pool_key = ("rival_a" if value[-1:] in "a13579" else "rival_b") if "rival" in value else "own" if "own" in value or "sample" in value else "default"
        topics = TOPIC_POOLS[pool_key]
        cid = "UC" + hashlib.sha256(value.encode()).hexdigest()[:22]
        subs = rng.randint(8_000, 400_000)
        name = f"[SAMPLE] {ref.value.lstrip('@').title()}"
        videos = []
        n_total = max(sample_size, 12)
        for i in range(n_total):
            topic = rng.choice(topics)
            title = rng.choice(TEMPLATES).format(t=topic).capitalize()
            age = int(i * rng.uniform(5, 12) + rng.uniform(1, 6))
            pub = self.now - timedelta(days=age, hours=rng.randint(0, 23))
            if date_from and pub < date_from or date_to and pub > date_to:
                continue
            short = rng.random() < 0.3
            views = int(rng.lognormvariate(8.6 + (0.5 if topic in ("espresso", "cold brew") else 0), 1.0))
            vid = hashlib.sha256(f"{cid}{i}".encode()).hexdigest()[:11]
            videos.append({
                "video_id": vid, "title": title, "url": f"https://www.youtube.com/watch?v={vid}",
                "published_at": pub.isoformat().replace("+00:00", "Z"),
                "duration_seconds": rng.randint(20, 58) if short else rng.randint(240, 1500),
                "views": views, "likes": int(views * rng.uniform(0.02, 0.06)) if rng.random() > 0.05 else None,
                "comments": int(views * rng.uniform(0.001, 0.008)), "description": f"[SAMPLE] Illustrative description about {topic}.",
                "tags": [topic, "coffee"], "thumbnail_url": None, "category": "Howto & Style",
                "channel_id": cid, "channel_title": name,
            })
            if len(videos) >= sample_size:
                break
        channel = {
            "channel_id": cid, "title": name, "description": "[SAMPLE] Illustrative channel - not real data.",
            "custom_url": ref.value, "created_at": (self.now - timedelta(days=rng.randint(700, 2500))).isoformat(),
            "country": None, "subscribers": subs, "subscribers_hidden": False,
            "total_views": int(subs * rng.uniform(40, 120)), "total_videos": n_total + rng.randint(0, 80),
            "uploads_playlist": None, "thumbnail_url": None, "url": f"https://www.youtube.com/channel/{cid}",
        }
        return {"channel": channel, "videos": videos, "notes": ["SAMPLE DATA - illustrative only, not real YouTube data."]}

    def find_channel_candidates(self, queries: list[str], max_per_query: int = 50, published_after=None) -> list[dict]:
        """Illustrative stand-in for competitor discovery: 14 fake channels with declining 'hits'."""
        return [{"channel_id": f"rival_{i:02d}", "title": f"[SAMPLE] Rival_{i:02d}", "hits": max(1, 14 - i)} for i in range(1, 15)]

    def get_channels_by_ids(self, channel_ids: list[str]) -> list[dict]:
        out = []
        for cid in channel_ids:
            ch = self.fetch_channel_bundle(ParsedChannel("handle", "@" + cid), 12)["channel"]
            ch["_query_id"] = cid
            out.append(ch)
        return out
