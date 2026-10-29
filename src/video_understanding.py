"""Automatic understanding of an uploaded video: a few frames (+ optional transcript) go to the configured AI provider,
which returns the topic, a summary and a first draft of titles, description, tags and music mood.

Rules given to the AI: describe only what is visible/audible, do not identify real people from their faces, do not invent
facts, results or numbers, and flag possible rights issues (broadcast graphics, logos, watermarks, lyrics) as UNCERTAIN.
Without an AI key nothing is guessed: the app uses only the text the user typed.
"""
from __future__ import annotations

from typing import Any

SYSTEM = ("You are a careful YouTube content analyst. You receive frames sampled evenly from one video and may receive its transcript. "
          "Describe only what you can see or read; do not identify real people from their faces; never invent facts, scores, results or numbers. "
          "Flag possible rights issues (broadcast graphics, channel logos, watermarks, on-screen lyrics) as uncertain observations. "
          "Return ONLY JSON.")
SCHEMA = ('{"topic": "short phrase", "niche": "one or two words", "summary": "2 sentences", "key_moments": ["...", "..."], "mood": "...", '
          '"language": "...", "audience": "who would watch this", "title_ideas": ["5 titles under 70 characters"], '
          '"description": "a draft description without invented facts", "tags": ["8-12 tags"], "hashtags": ["3-6 hashtags without #"], '
          '"music_mood": "what music mood would fit", "rights_flags": ["uncertain observations, or empty"], "confidence": "low|medium|high"}')


def _strs(v: Any, limit: int, n: int) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip()[:limit] for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:n]


def _clean(d: Any) -> dict:
    if not isinstance(d, dict):
        raise ValueError("The AI response was not a JSON object.")
    conf = str(d.get("confidence", "low")).lower()
    return {"topic": str(d.get("topic", "")).strip()[:120], "niche": str(d.get("niche", "")).strip()[:40], "summary": str(d.get("summary", "")).strip()[:600],
            "key_moments": _strs(d.get("key_moments"), 160, 6), "mood": str(d.get("mood", "")).strip()[:80], "language": str(d.get("language", "")).strip()[:40],
            "audience": str(d.get("audience", "")).strip()[:160], "title_ideas": _strs(d.get("title_ideas"), 100, 8),
            "description": str(d.get("description", "")).strip()[:2000], "tags": _strs(d.get("tags"), 40, 15),
            "hashtags": [h.lstrip("#").replace(" ", "") for h in _strs(d.get("hashtags"), 30, 8)], "music_mood": str(d.get("music_mood", "")).strip()[:160],
            "rights_flags": _strs(d.get("rights_flags"), 200, 5), "confidence": conf if conf in ("low", "medium", "high") else "low"}


def understand(provider, frames: list[bytes], probe: dict | None, user_text: str = "", transcript: str | None = None, niche_hint: str = "") -> dict:
    """Ask the AI provider to understand the video. Raises on failure so the caller can fall back to the user's text."""
    if provider is None or not getattr(provider, "is_configured", False):
        raise ValueError("No AI provider configured.")
    if not frames:
        raise ValueError("No frames could be extracted from the video.")
    facts = []
    if probe:
        facts.append(f"duration {probe.get('duration_seconds', 0):.0f}s, {probe.get('width')}x{probe.get('height')} {probe.get('orientation', '')}")
    prompt = (f"Video facts: {'; '.join(facts) or 'n/a'}\nChannel niche hint: {niche_hint or 'unknown'}\nCreator's own notes: {user_text[:500] or 'none'}\n"
              f"Transcript (may be partial): {(transcript or 'not available')[:2500]}\n\nReturn JSON exactly like: {SCHEMA}")
    out = _clean(provider.complete_json(SYSTEM, prompt, max_tokens=1400, images=frames[:8]))
    out["source"] = "AI: " + ("frames + transcript" if transcript else "frames")
    return out


def fallback(user_title: str = "", user_topic: str = "", user_summary: str = "") -> dict:
    """No AI: use only what the user typed. Nothing about the video itself is inferred."""
    return {"topic": user_topic or user_title, "niche": "", "summary": user_summary, "key_moments": [], "mood": "", "language": "", "audience": "",
            "title_ideas": [], "description": "", "tags": [], "hashtags": [], "music_mood": "", "rights_flags": [], "confidence": "none",
            "source": "Your text only (no AI configured, so the video content was not analysed)"}
