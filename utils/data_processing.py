"""Shared helpers: durations, text tokenising, DataFrame conversion, JSON/CSV serialisation."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

VIDEO_COLUMNS = [
    "video_id", "title", "url", "published_at", "duration_seconds", "views", "likes", "comments",
    "description", "tags", "thumbnail_url", "category", "channel_id", "channel_title",
]

STOPWORDS = set(
    """a an and are as at be but by for from has have how i if in into is it its my of on or our
    so that the their then there these they this to was we were what when where which who why will
    with you your yours not no yes can do does did just about all any more most new now out up
    than too very get got via vs part ep episode video videos official full hd watch best top
    one two three using use make made makes need needs here every really been being them those
    his her him she he us me am also over after before while like still only own same such""".split()
)

_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'+#-]*")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
]


def parse_iso_duration(value: str | None) -> int:
    """Parse an ISO-8601 duration such as PT1H2M3S into seconds. Unknown input -> 0."""
    if not value:
        return 0
    m = _ISO_DURATION.match(value)
    if not m:
        return 0
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    v = float(value)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.1f}{unit}"
    return f"{v:,.0f}" if v == int(v) else f"{v:,.1f}"


def safe_median(series: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna()
    return float(arr.median()) if len(arr) else float("nan")


def tokenize(text: str | None) -> list[str]:
    """Lower-case content words (stopwords/short tokens removed)."""
    if not text:
        return []
    text = _URL_RE.sub(" ", text)
    out = []
    for tok in _TOKEN_RE.findall(text):
        t = tok.lower().strip("'-")
        if len(t) < 3 or t in STOPWORDS:
            continue
        out.append(t)
    return out


_SEGMENT_SPLIT = re.compile(r"[.,;:!?|\n()\[\]/]|\s[-\u2013\u2014]\s")


def term_set(text: str | None) -> set[str]:
    """Unigrams plus bigrams of ADJACENT content words (bigrams never span punctuation or stopwords)."""
    if not text:
        return set()
    terms: set[str] = set()
    for segment in _SEGMENT_SPLIT.split(_URL_RE.sub(" ", text)):
        raw = [t.lower().strip("'-") for t in _TOKEN_RE.findall(segment)]
        keep = [t if (len(t) >= 3 and t not in STOPWORDS) else None for t in raw]
        terms.update(t for t in keep if t)
        terms.update(f"{a} {b}" for a, b in zip(keep, keep[1:]) if a and b)
    return terms


def videos_to_df(videos: list[dict]) -> pd.DataFrame:
    """Normalised video dicts -> typed DataFrame (possibly empty)."""
    if not videos:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in VIDEO_COLUMNS}).assign(
            published_at=pd.Series(dtype="datetime64[ns, UTC]")
        )
    df = pd.DataFrame(videos)
    for col in VIDEO_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[VIDEO_COLUMNS].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype("int64")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).astype("int64")
    for col in ("likes", "comments"):
        df[col] = pd.to_numeric(df[col], errors="coerce")  # NaN = hidden / disabled / unavailable
    df["tags"] = df["tags"].apply(lambda t: list(t) if isinstance(t, (list, tuple)) else [])
    for col in ("title", "description"):
        df[col] = df[col].fillna("").astype(str)
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- serialisation


def redact_secrets(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def to_jsonable(obj: Any) -> Any:
    """Recursively convert DataFrames, numpy/pandas scalars, NaN and datetimes into JSON-safe values."""
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        if obj is pd.NaT:
            return None
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def results_to_json_bytes(results: dict) -> bytes:
    """Serialise analysis results (never any credentials) to UTF-8 JSON."""
    payload = to_jsonable({k: v for k, v in results.items() if not k.startswith("_")})
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return redact_secrets(text).encode("utf-8")


def prepare_df_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for CSV/Excel/Word/PDF: naive datetimes, flat lists, no NaN objects."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif s.dtype == object or str(s.dtype).startswith("str"):
            out[col] = s.apply(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v)
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel opens accented characters correctly."""
    return prepare_df_for_export(df).to_csv(index=False).encode("utf-8-sig")


def top_terms(texts: Iterable[str], n: int = 10) -> list[tuple[str, int]]:
    """Most common unigrams by document frequency."""
    c: Counter[str] = Counter()
    for t in texts:
        c.update(set(tokenize(t)))
    return c.most_common(n)


# ----------------------------------------------------------------------------- export tables

VIDEO_EXPORT_COLS = {
    "title": "Title", "url": "Video URL", "published_at": "Published (UTC)", "duration_seconds": "Duration (s)",
    "views": "Views (public)", "likes": "Likes (public)", "comments": "Comments (public)", "views_per_day": "Views per day",
    "relative_to_median": "Relative to median", "engagement_rate": "Engagement rate", "format": "Format",
    "topic": "Topic", "performance_flag": "Performance flag", "category": "Category", "tags": "Tags",
}
TITLE_EXPORT_COLS = {
    "title": "Title", "url": "Video URL", "views": "Views (public)", "relative_to_median": "Relative to median",
    "title_chars": "Characters", "title_words": "Words", "title_has_number": "Has number", "title_has_question": "Question",
    "title_pattern": "Pattern", "title_length_bucket": "Length bucket", "keyword_in_first_40": "Keyword in first 40 chars",
    "wording_flags": "Wording to review",
}


def _pick(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    cols = [c for c in mapping if c in df.columns]
    return df[cols].rename(columns=mapping)


def build_export_tables(results: dict) -> dict[str, pd.DataFrame]:
    """Relevant sheets only. Keys are the Excel sheet names (also used for CSV/Word/PDF section titles)."""
    t: dict[str, pd.DataFrame] = {}
    if len(results.get("overview_table", [])):
        t["Channel Overview"] = results["overview_table"]
    videos = results.get("videos")
    if videos is not None and len(videos):
        t["Video Performance"] = _pick(videos, VIDEO_EXPORT_COLS)
        t["Title Analysis"] = _pick(videos, TITLE_EXPORT_COLS)
    if len(results.get("title_patterns", [])):
        t["Title Patterns"] = results["title_patterns"].rename(columns=lambda c: c.replace("_", " ").title())
    if len(results.get("topics", [])):
        t["Topic Analysis"] = results["topics"].rename(columns=lambda c: c.replace("_", " ").title())
    if len(results.get("competitors", [])):
        t["Competitors"] = results["competitors"].rename(columns=lambda c: c.replace("_", " ").title())
    if len(results.get("gaps", [])):
        t["Content Gaps"] = results["gaps"].rename(columns=lambda c: c.replace("_", " ").title())
    if len(results.get("calendar", [])):
        t["Content Calendar"] = results["calendar"].rename(columns=lambda c: c.replace("_", " ").title())
    rm = results.get("roadmap") or {}
    if rm.get("available"):
        k = rm["kpis"][["metric", "you", "benchmark", "benchmark_source", "attainment", "status", "target_30", "target_60", "target_90"]].copy()
        for c in ("you", "benchmark", "attainment", "target_30", "target_60", "target_90"):
            k[c] = k[c].astype(float).round(4)
        t["Growth Roadmap"] = k.rename(columns={"metric": "Metric", "you": "You", "benchmark": "Benchmark", "benchmark_source": "Benchmark source",
                                                "attainment": "Attainment vs benchmark", "status": "Status", "target_30": "Day 30 target",
                                                "target_60": "Day 60 target", "target_90": "Day 90 target"})
        t["Roadmap Actions"] = rm["actions"].rename(columns=lambda c: c.replace("_", " ").title())
        t["Daily Plan"] = rm["daily_plan"].rename(columns=lambda c: c.replace("_", " ").title())
    mo = results.get("monetization") or {}
    if mo.get("available"):
        b, p = mo["baseline"], mo["potential_monthly_income"]
        el = mo["eligible_now"]
        rows = [("Monetization status (your choice)", mo["status"]), ("Subscribers used", round(b["subs"])),
                ("Estimated public long-form watch hours, 12 months", round(b["hours_12m"])), ("Estimated Shorts views, 90 days", round(b["shorts_90d"])),
                ("Meets ads tier by estimate", "Yes" if el["ads"] else "No"), ("Treated as earning now", "Yes" if mo["earning_now"] else "No"),
                ("Estimated monthly income if monetized, low (USD)", round(p["low"], 2)), ("Estimated monthly income if monetized, mid (USD)", round(p["mid"], 2)),
                ("Estimated monthly income if monetized, high (USD)", round(p["high"], 2)),
                ("Estimate caveat", "Estimate from public data and assumptions; real revenue is private to the owner (YouTube Studio).")]
        t["Income Estimate"] = pd.DataFrame(rows, columns=["Metric", "Value"])
        t["Income Roadmap"] = mo["steps"].rename(columns=lambda c: c.replace("_", " ").title())
        t["Income Timeline"] = mo["eligibility"].rename(columns=lambda c: c.replace("_", " ").title())
        t["Income Projection"] = mo["projection"].rename(columns=lambda c: c.replace("_", " ").title())
        t["Income Notes"] = pd.DataFrame({"Note": mo["notes"]})
    if len(results.get("discovered_competitors", [])):
        t["Auto Competitors"] = results["discovered_competitors"].drop(columns=["channel_id"], errors="ignore").rename(columns=lambda c: c.replace("_", " ").title())
    meta = results.get("meta", {})
    rows = [{"Type": "Analysis", "Item": "Analysis date (UTC)", "Detail": meta.get("analysis_date")},
            {"Type": "Analysis", "Item": "Data mode", "Detail": meta.get("mode")},
            {"Type": "Analysis", "Item": "Topic method", "Detail": meta.get("topic_method")}]
    rows += [{"Type": s["type"], "Item": s["item"], "Detail": s["detail"]} for s in results.get("sources", [])]
    rows += [{"Type": "Limitation", "Item": f"{i + 1}", "Detail": lim} for i, lim in enumerate(results.get("limitations", []))]
    rows += [{"Type": "Note", "Item": "", "Detail": n} for n in results.get("notes", [])]
    rows += [{"Type": "Warning", "Item": "", "Detail": w} for w in results.get("warnings", [])]
    t["Sources and Notes"] = pd.DataFrame(rows)
    return t
