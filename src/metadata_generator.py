"""Title, caption (description) and hashtag generator.

Candidates are ranked by a transparent 'channel fit' score built from this channel's own public results (which title patterns and lengths
went with more views). The score is a way to sort options, NOT a prediction of views. Titles never invent numbers or claims: a number
appears only when it matches the key points the creator supplied.
"""
from __future__ import annotations

import re

import pandas as pd

from src.content_strategy import CTAS
from src.title_analysis import _length_bucket, classify_pattern, wording_flags

STOP = {"the", "a", "an", "of", "and", "to", "in", "for", "with", "on", "is", "my", "your"}


def _title_case(s: str) -> str:
    return " ".join(w if (w.isupper() and len(w) > 1) else w.capitalize() if w.lower() not in STOP else w.lower() for w in s.split()).strip()


def _hashtag(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9\u00C0-\uFFFF]+", text)
    return "#" + "".join(w.capitalize() for w in words)[:30] if words else ""


def candidate_titles(topic: str, key_points: list[str], fmt: str, duration_s: float | None, other: str = "") -> list[str]:
    t, T = topic.strip().lower(), _title_case(topic.strip())
    out = [f"How to {t}", f"How to {t}: step by step", f"{T}: a beginner's guide", f"Is {t} worth it?", f"How does {t} work?",
           f"{T} explained", f"{T}: what I learned", f"{T}: what you need to know"]
    if len(key_points) >= 2:
        out.append(f"{len(key_points)} things to know about {t}")
    if other.strip():
        out.append(f"{T} vs {_title_case(other.strip())}: which is better?")
    if fmt == "short":
        out = [f"{T} in {int(duration_s)} seconds" if duration_s and duration_s <= 60 else f"{T} in under a minute", f"One {t} tip", f"{T}: quick answer",
               f"How to {t}", f"Is {t} worth it?", f"{T} explained fast"] + out[-2:]
    seen, uniq = set(), []
    for x in out:
        x = re.sub(r"\s+", " ", x).strip()
        if x.lower() not in seen:
            seen.add(x.lower())
            uniq.append(x[0].upper() + x[1:])
    return uniq


def score_title(title: str, profile: dict, keyword: str, fmt: str) -> tuple[int, list[str]]:
    score, why = 50, ["base 50"]
    pat = classify_pattern(title)
    eff = (profile.get("effects") or {}).get("title_pattern")
    if eff is not None and len(eff):
        row = eff[eff["group"] == pat]
        if len(row):
            pts = int(max(-20, min(20, (float(row["effect"].iloc[0]) - 1.0) * 40)))
            score += pts
            why.append(f"{pat} pattern: {pts:+d} (your videos with this pattern: {float(row['median_relative'].iloc[0]):.2f}x typical, {int(row['videos'].iloc[0])} videos)")
    best_len = (profile.get("best") or {}).get("title_length_bucket")
    bucket = _length_bucket(len(title))
    if best_len and bucket == best_len:
        score += 10
        why.append(f"length {bucket} matches your best band +10")
    elif fmt == "short" and len(title) <= 45:
        score += 5
        why.append("short title suits mobile Shorts feed +5")
    if len(title) > 70:
        score -= 10
        why.append("over 70 characters may be truncated -10")
    pos = title.lower().find(keyword.lower()) if keyword else -1
    if 0 <= pos <= 25:
        score += 10
        why.append("topic keyword near the start +10")
    elif 0 <= pos <= 40:
        score += 5
        why.append("topic keyword in first 40 characters +5")
    for f in wording_flags(title):
        score -= 15
        why.append(f"wording to review ({f}) -15")
    return max(0, min(100, score)), why


def hashtags_for(topic: str, keywords: list[str], profile: dict, fmt: str, limit: int = 5) -> list[str]:
    cands = [_hashtag(topic)] + [_hashtag(k) for k in keywords] + [f"#{h}" for h in profile.get("hashtags_used", [])[:4]]
    if fmt == "short":
        cands.append("#Shorts")
    seen, out = set(), []
    for c in cands:
        if c and c.lower() not in seen and len(c) > 2:
            seen.add(c.lower())
            out.append(c)
    return out[:limit]


def tags_for(topic: str, keywords: list[str], profile: dict, max_chars: int = 450) -> list[str]:
    cands = [topic.lower()] + [k.lower() for k in keywords] + [f"how to {topic.lower()}"] + list(profile.get("tags_used", []))[:8]
    out, total, seen = [], 0, set()
    for c in cands:
        c = c.strip()
        if c and c not in seen and total + len(c) + 1 <= max_chars:
            seen.add(c)
            out.append(c)
            total += len(c) + 1
    return out


def build_description(topic: str, key_points: list[str], fmt: str, duration_s: float | None, goal: str, hashtags: list[str], audience: str = "") -> str:
    who = f" for {audience.strip()}" if audience.strip() else ""
    cta = CTAS.get(goal.lower(), CTAS["subscribers"])
    if fmt == "short":
        pts = key_points[0] if key_points else f"the key idea about {topic}"
        return f"{_title_case(topic)}: {pts}.\n\n{cta}\n\n{' '.join(hashtags)}"
    lines = [f"In this video: {topic}{who}. Here is what you will get and why it matters.", ""]
    lines += [f"- {p}" for p in key_points] or ["- (add the 3 main things viewers will learn)"]
    if duration_s and duration_s >= 480:
        lines += ["", "Chapters (replace with your real timestamps; the first must be 00:00):", "00:00 Intro", "00:00 (add each section)"]
    lines += ["", cta, "", "Links and credits: add only real links; disclose sponsorships/affiliate links; add each music track's required attribution.", "", " ".join(hashtags)]
    return "\n".join(lines)


def _ai_titles(provider, topic: str, key_points: list[str], fmt: str, profile: dict, audience: str) -> list[str]:
    system = ("You write accurate YouTube titles. Keep the subject. Never invent numbers, results or claims beyond the key points. No clickbait. Return ONLY JSON.")
    best = (profile.get("best") or {})
    prompt = (f"Topic: {topic}\nFormat: {fmt}\nAudience: {audience or 'unspecified'}\nKey points: {key_points}\n"
              f"Channel's best-performing title pattern: {best.get('title_pattern')}; best title length: {best.get('title_length_bucket')}.\n"
              'Return a JSON list of 6 strings (titles only), each under 70 characters.')
    data = provider.complete_json(system, prompt, max_tokens=500)
    return [str(x).strip() for x in data if isinstance(x, (str, int)) and str(x).strip()][:6]


def generate(profile: dict, topic: str, keywords: list[str] | None = None, key_points: list[str] | None = None, fmt: str = "long",
             duration_s: float | None = None, goal: str = "subscribers", audience: str = "", other: str = "", provider=None) -> dict:
    if not topic.strip():
        raise ValueError("Enter the video's topic.")
    keywords = [k.strip() for k in (keywords or []) if k.strip()]
    key_points = [p.strip() for p in (key_points or []) if p.strip()]
    cands = [(t, "Template") for t in candidate_titles(topic, key_points, fmt, duration_s, other)]
    source_note = "Rule-based from your channel's patterns."
    if provider is not None and getattr(provider, "is_configured", False):
        try:
            cands = [(t, "AI-assisted") for t in _ai_titles(provider, topic, key_points, fmt, profile, audience)] + cands
            source_note = "AI-assisted wording, scored with your channel's patterns."
        except Exception:  # noqa: BLE001 - any provider failure falls back to rules
            source_note = "AI unavailable; rule-based titles shown."
    rows = []
    for title, src in cands:
        sc, why = score_title(title, profile, keywords[0] if keywords else topic.split()[0], fmt)
        rows.append({"title": title, "channel_fit_score": sc, "characters": len(title), "source": src, "score_breakdown": "; ".join(why)})
    titles = pd.DataFrame(rows).drop_duplicates("title").sort_values("channel_fit_score", ascending=False).head(8).reset_index(drop=True)
    tags = hashtags_for(topic, keywords, profile, fmt)
    return {"titles": titles, "description": build_description(topic, key_points, fmt, duration_s, goal, tags, audience), "hashtags": tags,
            "tags": tags_for(topic, keywords, profile), "pinned_comment": (f"Question for you: which part of {topic} do you find hardest? Tell me and I will cover it next."),
            "note": source_note + " The score sorts options by your own past results; it does not predict views. Only the first 3 hashtags show above the title, "
                    "and more than 15 can cause YouTube to ignore them all, so keep them few and relevant."}
