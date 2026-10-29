"""Title, caption and hashtag generator tuned to YOUR channel's observed patterns.

The 'channel fit' score measures how closely a draft matches what has worked on your channel so far (title pattern, length,
keyword placement, specificity) and flags risky wording or near-duplicates. It is NOT a prediction of views.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from src.title_analysis import classify_pattern, wording_flags
from utils.data_processing import tokenize

LENGTH_CENTRE = {"<30 chars": 25, "30-50 chars": 40, "51-70 chars": 60, ">70 chars": 75}
SHRINK_K = 3.0  # pull small-sample pattern effects toward 1.0 (no effect)
MAX_HASHTAGS = 15  # YouTube ignores ALL hashtags on a video if it has more than 15


def _words(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip())


def candidate_titles(topic: str, summary: str = "", fmt: str = "Long-form", n_hint: int = 5) -> list[str]:
    t = _words(topic)
    cap = t[:1].upper() + t[1:]
    short = fmt == "Short"
    c = [f"How to {t}", f"How to {t} (Step by Step)", f"{n_hint} Tips for {t}", f"{cap}: What to Know Before You Start",
         f"Is {t} Worth It?", f"{cap} for Beginners", f"The Simple Way to {t}", f"Common {t} Mistakes (and How to Fix Them)",
         f"My {t} Routine", f"{cap} Explained in Plain English"]
    if short:
        c = [f"{cap} in Under a Minute", f"How to {t} #shorts".replace(" #shorts", ""), f"{n_hint} Quick {t} Tips", f"Stop Doing This With {t}",
             f"{cap}: One Thing to Try Today", f"Is {t} Worth It?"] + c[:4]
    if summary.strip():
        first = re.split(r"[.!?\n]", summary.strip())[0].strip()
        if 12 <= len(first) <= 70:
            c.append(first[:1].upper() + first[1:])
    seen, out = set(), []
    for x in c:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def score_title(title: str, ctx: dict, topic_words: set[str] | None = None) -> dict:
    """0-100 channel-fit score with a component breakdown."""
    topic_words = topic_words or set()
    pat = classify_pattern(title)
    perf = ctx.get("pattern_perf", {}).get(pat)
    if perf and perf[1] >= 1:
        rel, n = perf
        fit = (n * rel + SHRINK_K * 1.0) / (n + SHRINK_K)
        pattern_pts = 30 * min(1.0, max(0.0, (fit - 0.6) / 1.0))
        pat_note = f"'{pat}' titles: {n} on your channel at {rel:.2f}x median (shrunk to {fit:.2f}x)"
    else:
        pattern_pts, pat_note = 15.0, f"No channel evidence for '{pat}' titles (neutral)"
    ideal = LENGTH_CENTRE.get(ctx.get("best_length_bucket") or "", 50)
    length_pts = 20 * max(0.0, 1 - abs(len(title) - ideal) / 30)
    kws = set(ctx.get("keywords", [])) | set(ctx.get("trending_terms", [])) | topic_words
    low = title.lower()
    first = low[:40]
    if any(k in first for k in kws):
        kw_pts, kw_note = 20.0, "keyword in the first 40 characters"
    elif any(k in low for k in kws):
        kw_pts, kw_note = 10.0, "keyword present but late in the title"
    else:
        kw_pts, kw_note = 0.0, "no channel/topic keyword"
    spec_pts = (8 if re.search(r"\d", title) else 0) + (7 if len(title.split()) >= 5 else 0)
    flags = wording_flags(title)
    own = ctx.get("own_titles", [])
    dup = max((SequenceMatcher(None, low, o.lower()).ratio() for o in own), default=0.0)
    safety = 15.0 - (8 if flags else 0) - (15 if dup >= 0.8 else 0)
    total = max(0.0, pattern_pts + length_pts + kw_pts + spec_pts + safety)
    notes = [pat_note, f"{len(title)} characters (your best-performing length is about {ideal})", kw_note]
    if flags:
        notes.append("wording to review: " + "; ".join(flags))
    if dup >= 0.8:
        notes.append(f"{dup:.0%} similar to an existing title of yours")
    if len(title) > 70:
        notes.append("may be truncated in search and mobile feeds")
    return {"title": title, "pattern": pat, "fit_score": round(total, 1), "pattern_pts": round(pattern_pts, 1), "length_pts": round(length_pts, 1),
            "keyword_pts": kw_pts, "specificity_pts": spec_pts, "safety_pts": max(safety, 0), "notes": "; ".join(notes)}


def rank_titles(topic: str, ctx: dict, summary: str = "", fmt: str = "Long-form", extra: list[str] | None = None, top: int = 8, keep_ai: int = 3) -> pd.DataFrame:
    """Score template titles and any AI titles. The best `keep_ai` AI titles are always shown (they are based on the actual video)."""
    tw = set(tokenize(topic))
    ai_set = {e for e in (extra or []) if e}
    rows = [score_title(c, ctx, tw) for c in candidate_titles(topic, summary, fmt) + list(ai_set)]
    df = pd.DataFrame(rows).drop_duplicates("title")
    df["source"] = df["title"].apply(lambda t: "AI" if t in ai_set else "Template")
    ai = df[df["source"] == "AI"].sort_values("fit_score", ascending=False).head(keep_ai)
    rest = df[~df["title"].isin(ai["title"])].sort_values("fit_score", ascending=False).head(max(top - len(ai), 0))
    return pd.concat([ai, rest]).sort_values("fit_score", ascending=False).reset_index(drop=True)


def _clean_tag(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "", s.title().replace(" ", ""))


def hashtags(topic: str, ctx: dict, fmt: str = "Long-form", limit: int = 8) -> dict:
    """Rank candidate hashtags; the first 3 are the ones YouTube shows above the title."""
    topic_toks = tokenize(topic)
    scored: dict[str, float] = {}

    def add(raw: str, w: float):
        tag = _clean_tag(raw)
        if 2 < len(tag) <= 30:
            scored[tag] = scored.get(tag, 0) + w

    if len(topic_toks) >= 2:
        add(" ".join(topic_toks[:3]), 4.0)
    for t in topic_toks:
        add(t, 3.0 if len(topic_toks) == 1 else 1.5)
    for i, tag in enumerate(ctx.get("top_tags", [])[:12]):
        add(tag, 2.0 - i * 0.1)
    for i, term in enumerate(ctx.get("trending_terms", [])[:8]):
        add(term, 1.2 - i * 0.05)
    for k in ctx.get("keywords", [])[:4]:
        add(k, 1.0)
    ordered = [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])]
    if fmt == "Short" and "Shorts" not in ordered:
        ordered.insert(min(3, len(ordered)), "Shorts")
    chosen = ordered[:min(limit, MAX_HASHTAG_SAFE)]
    plain = [f"#{t}" for t in chosen]
    hidden = ", ".join(dict.fromkeys([topic.strip().lower()] + [t.lower() for t in ctx.get("top_tags", [])[:8]]))[:480]
    return {"hashtags": plain, "above_title": plain[:3], "hidden_tags": hidden,
            "note": f"YouTube shows the first 3 above the title and ignores ALL hashtags if a video has more than {MAX_HASHTAGS}. Keep them relevant; irrelevant tags can be treated as spam."}


MAX_HASHTAG_SAFE = 10


def caption(topic: str, title: str, summary: str, fmt: str, tags: list[str], goal: str = "subscribers", music_credit: str = "") -> str:
    s = summary.strip() or "[Write 2 sentences: what viewers will get from this video and one concrete detail.]"
    ctas = {"subscribers": "If this helped, subscribe for more.", "watch time": "Next, watch the related video linked below.",
            "engagement": "What would you like me to cover next? Tell me in the comments.", "traffic": "Link in the description."}
    cta = ctas.get(goal, ctas["subscribers"])
    tag_line = " ".join(tags[:MAX_HASHTAG_SAFE])
    if fmt == "Short":
        return f"{title}\n\n{s}\n\n{cta}\n\n{tag_line}"
    parts = [title, "", s, "", "Chapters:", "0:00 Intro", "[add timestamps for each section]", "", cta]
    if music_credit.strip():
        parts += ["", f"Music: {music_credit.strip()}"]
    else:
        parts += ["", "Music: [track] by [artist] - [licence / link]"]
    parts += ["", tag_line]
    return "\n".join(parts)


def ai_titles(provider, topic: str, summary: str, ctx: dict, fmt: str, n: int = 6) -> list[str]:
    """Optional: ask the AI provider for extra titles. Raises on failure so the UI can fall back to templates."""
    system = ("You are a careful YouTube title writer. Use only facts in the brief; never invent results, numbers or promises. "
              "No misleading clickbait. Return ONLY a JSON list of strings.")
    prompt = (f"Topic: {topic}\nSummary: {summary[:500]}\nFormat: {fmt}\nChannel niche: {ctx.get('niche')}\nBest-performing title pattern: {ctx.get('best_pattern')}\n"
              f"Channel keywords: {', '.join(ctx.get('keywords', [])[:8])}\nReturn {n} titles under 70 characters.")
    data = provider.complete_json(system, prompt, max_tokens=500)
    titles = [str(x).strip() for x in data if isinstance(x, str) and x.strip()]
    if not titles:
        raise ValueError("AI returned no usable titles.")
    return titles[:n]
