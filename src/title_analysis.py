"""Title (and basic thumbnail) analysis. Patterns are observational, not causal."""
from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import pandas as pd

from utils.data_processing import tokenize, top_terms

MIN_GROUP = 3  # groups smaller than this are flagged as low-sample

SENSATIONAL = [
    "you won't believe", "you wont believe", "shocking", "insane", "unbelievable", "secret", "exposed",
    "destroyed", "gone wrong", "100%", "guaranteed", "instantly", "overnight", "never seen", "banned",
    "they don't want", "changes everything", "will blow your mind", "worst ever", "best ever",
]
NUMBER_RE = re.compile(r"\d")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
QUESTION_STARTS = ("how", "why", "what", "when", "which", "can", "should", "is", "are", "does", "do", "will")


def classify_pattern(title: str) -> str:
    t = title.strip().lower()
    if re.match(r"^(top\s+)?\d+\b", t) or re.search(r"\b\d+\s+(ways|tips|things|reasons|mistakes|steps|best|facts|moments|goals)\b", t):
        return "Number / list"
    if t.startswith("how to") or t.startswith("how i") or "tutorial" in t or "guide" in t:
        return "How-to / guide"
    if t.endswith("?") or t.split(" ")[0] in QUESTION_STARTS:
        return "Question"
    if " vs " in f" {t} " or "versus" in t or "compar" in t:
        return "Comparison"
    if "review" in t or "unboxing" in t or "hands-on" in t:
        return "Review"
    return "Statement / other"


def _length_bucket(n: int) -> str:
    if n < 30:
        return "<30 chars"
    if n <= 50:
        return "30-50 chars"
    if n <= 70:
        return "51-70 chars"
    return ">70 chars"


def wording_flags(title: str) -> list[str]:
    """Wording worth reviewing for overpromising. These are prompts for human judgement, not verdicts."""
    flags = []
    low = title.lower()
    hits = [p for p in SENSATIONAL if p in low]
    if hits:
        flags.append("sensational wording: " + ", ".join(hits[:3]))
    caps = [w for w in re.findall(r"\b[A-Z]{3,}\b", title) if not w.isdigit()]
    if len(caps) >= 2:
        flags.append("multiple ALL-CAPS words")
    if title.count("!") >= 2 or title.count("?") >= 2:
        flags.append("repeated punctuation")
    return flags


def analyze_titles(df: pd.DataFrame, keyword_count: int = 10) -> pd.DataFrame:
    """Add title feature columns to a copy of df."""
    out = df.copy()
    if out.empty:
        return out
    keywords = [k for k, _ in top_terms(out["title"], keyword_count)]
    titles = out["title"].astype(str)
    out["title_chars"] = titles.str.len()
    out["title_words"] = titles.str.split().str.len()
    out["title_has_number"] = titles.apply(lambda t: bool(NUMBER_RE.search(t)))
    out["title_has_question"] = titles.apply(
        lambda t: "?" in t or (t.strip().lower().split(" ")[0] if t.strip() else "") in QUESTION_STARTS)
    out["title_has_brackets"] = titles.str.contains(r"[\(\[\{]", regex=True)
    out["title_has_emoji"] = titles.apply(lambda t: bool(EMOJI_RE.search(t)))
    out["title_pattern"] = titles.apply(classify_pattern)
    out["title_length_bucket"] = out["title_chars"].apply(_length_bucket)
    out["keyword_in_first_40"] = titles.apply(lambda t: any(k in t[:40].lower() for k in keywords))
    out["wording_flags"] = titles.apply(lambda t: "; ".join(wording_flags(t)))
    out.attrs["title_keywords"] = keywords
    return out


def title_pattern_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median performance by title feature. Small groups are labelled; correlation is not causation."""
    if df.empty or "title_pattern" not in df:
        return pd.DataFrame(columns=["feature", "group", "videos", "median_views", "median_relative", "note"])
    specs = [
        ("Title pattern", "title_pattern"), ("Title length", "title_length_bucket"),
        ("Contains a number", "title_has_number"), ("Is a question", "title_has_question"),
        ("Keyword in first 40 chars", "keyword_in_first_40"), ("Has brackets", "title_has_brackets"),
    ]
    rows = []
    for label, col in specs:
        for grp, sub in df.groupby(col):
            n = len(sub)
            rows.append({
                "feature": label, "group": str(grp), "videos": n,
                "median_views": float(sub["views"].median()),
                "median_relative": float(sub["relative_to_median"].median()) if "relative_to_median" in sub else np.nan,
                "note": "low sample - treat as anecdotal" if n < MIN_GROUP else "",
            })
    return pd.DataFrame(rows)


def similar_titles(df: pd.DataFrame, threshold: float = 0.6, limit: int = 15) -> pd.DataFrame:
    """Pairs of near-duplicate titles (token Jaccard blended with character similarity)."""
    titles = df["title"].tolist()
    toks = [set(tokenize(t)) for t in titles]
    rows = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            a, b = toks[i], toks[j]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac < threshold * 0.6:
                continue
            score = 0.5 * jac + 0.5 * SequenceMatcher(None, titles[i].lower(), titles[j].lower()).ratio()
            if score >= threshold:
                rows.append({"title_a": titles[i], "title_b": titles[j], "similarity": round(score, 2)})
    return pd.DataFrame(sorted(rows, key=lambda r: -r["similarity"])[:limit],
                        columns=["title_a", "title_b", "similarity"])


def repeated_structures(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """Most common opening two-word structures."""
    starts = df["title"].apply(lambda t: " ".join(t.lower().split()[:2]))
    vc = starts.value_counts()
    vc = vc[vc >= min_count]
    return pd.DataFrame({"opening": vc.index, "videos": vc.values})


# ----------------------------------------------------------------------------- suggestions

def _clean_title(title: str) -> str:
    t = EMOJI_RE.sub("", title)
    t = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], t)
    words = t.split()
    words = [w.capitalize() if (w.isupper() and len(w) >= 4 and w.isalpha()) else w for w in words]
    return re.sub(r"\s+", " ", " ".join(words)).strip(" -|")


def suggest_titles_rule_based(title: str, channel_keywords: list[str] | None = None) -> list[dict]:
    """Three conservative starting points that keep the video's subject unchanged."""
    cleaned = _clean_title(title)
    ideas: list[dict] = []
    ideas.append({"title": cleaned, "improvement": "Tidied wording: removes emoji, repeated punctuation and shouty capitals for readability."})
    kw = next((k for k in (channel_keywords or []) if k in cleaned.lower() and k not in cleaned.lower()[:25]), None)
    if kw:
        front = f"{kw.title()}: {re.sub(re.escape(kw), '', cleaned, count=1, flags=re.I).strip(' -:|,')}"
        ideas.append({"title": re.sub(r"\s+", " ", front), "improvement": f"Moves the main topic keyword ('{kw}') to the front so the subject is clear at a glance."})
    if len(cleaned) > 60:
        cut = cleaned[:60].rsplit(" ", 1)[0].rstrip(",:-")
        ideas.append({"title": cut, "improvement": "Shortened to about 60 characters to reduce truncation in search and mobile feeds."})
    ideas.append({"title": f"{cleaned} [add the concrete result or number, only if accurate]",
                  "improvement": "Prompt to add specificity: a real result, number or audience that the video actually delivers."})
    seen, uniq = set(), []
    for i in ideas:
        if i["title"].lower() not in seen and i["title"].lower() != title.lower():
            seen.add(i["title"].lower())
            uniq.append(i)
    return uniq[:3]


def suggest_titles_ai(provider, title: str, description: str = "", niche: str = "") -> list[dict]:
    """Ask the configured AI provider for 3 alternatives. Raises on failure so the UI can fall back."""
    system = ("You are a careful YouTube title editor. Keep the video's actual subject. Never invent claims, numbers or "
              "outcomes that the description does not support. No misleading clickbait. Return ONLY JSON.")
    prompt = (f"Current title: {title}\nDescription (truncated): {description[:600]}\nNiche: {niche or 'unspecified'}\n"
              'Return a JSON list of exactly 3 objects: [{"title": "...", "improvement": "one sentence on what is improved"}]')
    data = provider.complete_json(system, prompt, max_tokens=600)
    out = [{"title": str(d.get("title", "")).strip(), "improvement": str(d.get("improvement", "")).strip()}
           for d in data if isinstance(d, dict) and d.get("title")]
    if not out:
        raise ValueError("AI returned no usable titles.")
    return out[:3]


# ----------------------------------------------------------------------------- thumbnails

def thumbnail_stats(image_bytes: bytes) -> dict:
    """Simple image statistics from a public thumbnail. These do NOT predict click-through rate."""
    from PIL import Image  # imported lazily so the rest of the app works without Pillow

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((320, 180))
    rgb = np.asarray(img, dtype=float) / 255.0
    gray = rgb.mean(axis=2)
    hsv = np.asarray(img.convert("HSV"), dtype=float) / 255.0
    gy, gx = np.gradient(gray)
    edges = np.hypot(gx, gy)
    return {
        "brightness": round(float(gray.mean()), 3),
        "contrast": round(float(gray.std()), 3),
        "saturation": round(float(hsv[..., 1].mean()), 3),
        "visual_busyness": round(float((edges > 0.1).mean()), 3),
    }


def analyze_thumbnails(df: pd.DataFrame, fetch: Callable[[str], bytes], limit: int = 20) -> pd.DataFrame:
    """Compute image stats for up to `limit` thumbnails. `fetch(url)` must return bytes (raise on failure)."""
    rows = []
    for _, r in df.head(limit).iterrows():
        url = r.get("thumbnail_url")
        if not url:
            continue
        try:
            stats = thumbnail_stats(fetch(url))
        except Exception:  # noqa: BLE001 - one broken thumbnail must not stop the batch
            continue
        rows.append({"video_id": r["video_id"], "title": r["title"], "views": r["views"], **stats})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["note"] = "Image statistics only; says nothing about actual click-through rate."
    return res


def thumbnail_consistency(stats_df: pd.DataFrame) -> dict:
    """Spread of thumbnail statistics across videos (lower spread = more visually consistent)."""
    if stats_df.empty:
        return {}
    return {c: round(float(stats_df[c].std(ddof=0)), 3) for c in ("brightness", "contrast", "saturation", "visual_busyness")}
