"""Content calendar / plan generator. Every row is either tied to evidence or labelled a creative hypothesis."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FORMATS = ["YouTube Short", "Long-form video", "Educational video", "Tutorial", "Review / comparison",
           "Storytelling", "Commentary / analysis"]

HOOKS = {
    "YouTube Short": "Open with the single most useful or surprising point about {t} in the first 2 seconds, then deliver it in under 45 seconds.",
    "Long-form video": "Start with the problem viewers have with {t}, promise what they'll understand by the end, and keep that promise.",
    "Educational video": "Pose the one question about {t} that your viewers most often get wrong, then answer it step by step.",
    "Tutorial": "Show the finished result for {t} first, then say 'here is exactly how to get there'.",
    "Review / comparison": "Name the options for {t} and the one criterion you will judge them on before showing any results.",
    "Storytelling": "Open in the middle of a real moment related to {t}, then explain how you got there.",
    "Commentary / analysis": "State your position on {t} in one sentence and invite viewers to disagree in the comments.",
}
POINTS = {
    "YouTube Short": ["One idea only", "A concrete example or demo", "A reason to follow for the next one"],
    "Long-form video": ["Context and why it matters", "Main explanation with examples", "Mistakes to avoid", "Recap"],
    "Educational video": ["Core concept in plain language", "Worked example", "Common misconceptions", "Where to go next"],
    "Tutorial": ["What you need before starting", "Step-by-step walkthrough", "Troubleshooting", "Final result"],
    "Review / comparison": ["Criteria and how you tested", "Option-by-option findings", "Who each option suits", "Your verdict"],
    "Storytelling": ["Setup", "The turning point", "What you learned", "What the viewer can take from it"],
    "Commentary / analysis": ["What happened or what's claimed", "Your analysis with evidence", "Counter-arguments", "Your conclusion"],
}
CTAS = {
    "subscribers": "Invite viewers to subscribe if they want more on this topic.",
    "watch time": "Point to a related video as the natural next watch.",
    "engagement": "Ask one specific question and invite answers in the comments.",
    "brand awareness": "Remind viewers what the channel is about and where to find more.",
    "traffic": "Direct viewers to the link in the description (only if relevant and clearly disclosed).",
    "sales": "Mention the offer once, clearly disclosed, after delivering the value.",
}
FALLBACK_TOPICS = ["A beginner's guide to your niche", "Common mistakes in your niche", "Your most-asked viewer question answered",
                   "Behind the scenes of how you work", "A comparison of two popular approaches in your niche"]


def _post_offsets(days: int, posts_per_week: float) -> list[int]:
    count = max(1, round(days * posts_per_week / 7))
    count = min(count, days)
    step = days / count
    return sorted({min(days - 1, int(i * step)) for i in range(count)})


def _idea_pool(gaps: pd.DataFrame, topic_summary: pd.DataFrame | None, niche: str) -> list[dict]:
    pool: list[dict] = []
    for r in gaps.itertuples() if gaps is not None and len(gaps) else []:
        pool.append({"topic": r.topic, "title": r.suggested_title, "basis": f"Evidence ({r.gap_type}, priority {r.priority_score}, confidence {r.confidence})",
                     "format_hint": r.format})
    if topic_summary is not None and len(topic_summary):
        for r in topic_summary.itertuples():
            if "Strong" in r.observation:
                pool.append({"topic": f"{r.topic} - new angle", "title": f"{r.topic}: A New Angle",
                             "basis": f"Evidence (your '{r.topic}' videos performed above your median in this sample)", "format_hint": None})
    for t in FALLBACK_TOPICS:
        pool.append({"topic": (niche + ": " if niche else "") + t, "title": (niche + ": " if niche else "") + t,
                     "basis": "Creative hypothesis (no channel data behind this idea)", "format_hint": None})
    return pool


def generate_calendar(gaps: pd.DataFrame | None, topic_summary: pd.DataFrame | None = None, days: int = 7,
                      posts_per_week: float = 3, formats: list[str] | None = None, niche: str = "",
                      audience: str = "", goals: list[str] | None = None, language: str = "English",
                      start: date | None = None, provider=None) -> pd.DataFrame:
    """Rule-based calendar; if an AI provider is given its wording is merged in (and labelled)."""
    formats = [f for f in (formats or FORMATS[:3]) if f in FORMATS] or FORMATS[:1]
    goals = goals or ["subscribers"]
    start = start or date.today()
    pool = _idea_pool(gaps, topic_summary, niche)
    cta_text = " ".join(CTAS.get(g.lower(), "") for g in goals[:2]).strip() or CTAS["subscribers"]
    rows = []
    for n, offset in enumerate(_post_offsets(days, posts_per_week)):
        idea = pool[n % len(pool)]
        hint = idea["format_hint"]
        fmt = next((f for f in formats if hint and ((hint.startswith("Short") and f == "YouTube Short") or (hint.startswith("Long") and f != "YouTube Short"))), formats[n % len(formats)])
        d = start + timedelta(days=offset)
        follow = pool[(n + 1) % len(pool)]["topic"]
        rows.append({
            "date": d.isoformat(), "day": d.strftime("%a"), "topic": idea["topic"], "working_title": idea["title"], "format": fmt,
            "opening_hook": HOOKS[fmt].format(t=idea["topic"].lower()), "talking_points": " | ".join(POINTS[fmt]),
            "call_to_action": cta_text, "follow_up_topic": follow, "basis": idea["basis"],
            "target_audience": audience or "Not specified", "language": language, "source": "Rule-based",
        })
    cal = pd.DataFrame(rows)
    if provider is not None and getattr(provider, "is_configured", False) and not cal.empty:
        cal = _merge_ai(cal, provider, niche, audience, language)
    return cal


def _merge_ai(cal: pd.DataFrame, provider, niche: str, audience: str, language: str) -> pd.DataFrame:
    items = [{"i": int(i), "topic": r.topic, "format": r.format, "basis": r.basis} for i, r in cal.iterrows()]
    system = ("You are a careful YouTube content strategist. Keep each idea tied to its stated basis; do not invent statistics "
              "or promise views, subscribers or virality. Return ONLY JSON.")
    prompt = (f"Niche: {niche or 'unspecified'}; audience: {audience or 'unspecified'}; write in {language}.\nIdeas: {items}\n"
              'Return a JSON list: [{"i": <index>, "title": "...", "hook": "...", "points": "3-4 points separated by |"}]')
    try:
        data = provider.complete_json(system, prompt, max_tokens=2500)
    except Exception:  # noqa: BLE001 - keep rule-based calendar
        return cal
    out = cal.copy()
    for d in data if isinstance(data, list) else []:
        try:
            i = int(d["i"])
        except (KeyError, ValueError, TypeError):
            continue
        if i in out.index:
            if d.get("title"):
                out.at[i, "working_title"] = str(d["title"]).strip()
            if d.get("hook"):
                out.at[i, "opening_hook"] = str(d["hook"]).strip()
            if d.get("points"):
                out.at[i, "talking_points"] = str(d["points"]).strip()
            out.at[i, "source"] = "AI-assisted (recommendation, not a guaranteed improvement)"
    return out
