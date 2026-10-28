"""Editing and music advice for a new video, tuned to what has worked on THIS channel (public data) and to measurements of the uploaded file.

Music note: no public data says which song or genre makes a given audience watch longer, and YouTube does not expose music performance.
So the music plan is a production-standard recommendation matched to the channel's best-performing content style, with a built-in A/B
test so the channel can learn what its own viewers prefer from YouTube Studio retention data. No specific copyrighted songs are suggested.
"""
from __future__ import annotations

import pandas as pd

from src.audience_profile import SHORT_FMT, mature_with_format_relative

MOODS = {
    "Calm & focused": {"keys": ["how to", "tutorial", "guide", "learn", "beginner", "explained", "tips", "setup", "diy", "recipe", "study", "review"],
                       "genre": "Lo-fi, soft ambient, light acoustic or minimal electronic with no vocals", "bpm": (70, 100), "beats_per_cut": 8,
                       "mix": "Keep the music steady and low so every spoken word stays clear."},
    "Upbeat & energetic": {"keys": ["challenge", "vs", "best", "top", "fast", "insane", "epic", "workout", "game", "gaming", "speedrun", "trend", "hack", "wins"],
                           "genre": "Upbeat pop-instrumental, electronic, funk or hip-hop beats with a clear pulse and no vocals under speech", "bpm": (110, 140), "beats_per_cut": 4,
                           "mix": "Let the beat drive the cuts; drop the music out for one beat before a key line."},
    "Warm & cinematic": {"keys": ["story", "journey", "vlog", "travel", "day in", "behind", "documentary", "life", "history", "why i", "my"],
                         "genre": "Cinematic strings, soft piano or warm acoustic guitar that builds slowly", "bpm": (60, 95), "beats_per_cut": 8,
                         "mix": "Build the music toward the emotional point and resolve it with the ending."},
    "Light & neutral": {"keys": [], "genre": "Light corporate-pop, soft groove or gentle jazz instrumentals", "bpm": (85, 115), "beats_per_cut": 8,
                        "mix": "Neutral background; choose it so it never competes with the voice."},
}
CUT_TARGETS = {"short": (12.0, 30.0), "Calm & focused": (3.0, 8.0), "Upbeat & energetic": (8.0, 20.0), "Warm & cinematic": (4.0, 10.0), "Light & neutral": (4.0, 10.0)}
SOURCES = [
    "YouTube Audio Library / the Shorts sound library: made for use on YouTube; still read each track's terms (some require attribution or limit monetization).",
    "A paid music subscription whose licence explicitly covers monetized YouTube videos: keep the licence/receipt for each track.",
    "Music you create or commission: keep the written agreement.",
    "Avoid: popular commercial songs, 'no copyright' tracks with unclear licences, and 'free download' sites. A claim can block, mute or redirect revenue.",
]


def infer_mood(df: pd.DataFrame) -> tuple[str, str]:
    """Pick the style that matches the channel's best-performing videos (top quarter by relative views)."""
    d = mature_with_format_relative(df)
    if d.empty:
        return "Light & neutral", "No videos to learn from, so a neutral default is used."
    top = d[d["rel_fmt"] >= d["rel_fmt"].quantile(0.75)]
    score = dict.fromkeys(MOODS, 0.0)
    for r in top.itertuples():
        text = f"{r.title} {' '.join(r.tags or [])}".lower()
        for mood, spec in MOODS.items():
            score[mood] += sum(1 for k in spec["keys"] if k in text) * float(r.rel_fmt)
    mood = max(score, key=score.get)
    if score[mood] <= 0:
        return "Light & neutral", "Your best videos did not match a clear style, so a neutral default is used."
    return mood, f"Matched from the titles and tags of your {len(top)} best-performing videos (top quarter by views within format)."


def _row(area: str, advice: str, basis: str, priority: str = "Medium") -> dict:
    return {"area": area, "priority": priority, "advice": advice, "basis": basis}


def build_advice(fmt: str, duration_s: float | None, mood: str, profile: dict, media: dict | None = None) -> pd.DataFrame:
    is_short = fmt == "short"
    spec = MOODS.get(mood, MOODS["Light & neutral"])
    rows: list[dict] = []
    rows.append(_row("Hook", "Show the payoff or ask the question in the first 1-2 seconds; no logo or greeting first." if is_short else
                     "Within the first 15-30 seconds state what the viewer will get and show a glimpse of the result before any intro.",
                     "Standard retention practice; verify with the retention graph in YouTube Studio.", "High"))
    best = (profile.get("best") or {}).get("duration_bucket")
    if duration_s and best and profile.get("available"):
        eff = profile["effects"]["duration_bucket"]
        cur = pd.cut([duration_s], bins=[-1, 60, 180, 600, 1200, float("inf")], labels=["<=1 min", "1-3 min", "3-10 min", "10-20 min", "20+ min"])[0]
        row = eff[eff["group"] == str(cur)]
        if str(cur) == best:
            rows.append(_row("Length", f"Your {duration_s:.0f}s video sits in your best-performing length band ({best}).", "Your public views by length band.", "Low"))
        else:
            e = f"{float(row['effect'].iloc[0]):.2f}x" if len(row) else "no data"
            rows.append(_row("Length", f"Your best-performing band is {best}; this video is {cur} (relative effect {e}). Trim or extend only if the story allows it.",
                             f"Median views by length band on your channel ({profile['reliability']} reliability).", "Medium"))
    lo, hi = CUT_TARGETS["short" if is_short else mood]
    if media and media.get("available") and media.get("cuts_per_minute") is not None:
        cpm = media["cuts_per_minute"]
        verdict = "within" if lo <= cpm <= hi else ("below" if cpm < lo else "above")
        tip = {"within": "Pacing fits the target.", "below": "Add cut-aways, zooms, B-roll or on-screen text to keep visual change flowing.",
               "above": "Let some shots breathe; very fast cutting can tire viewers on explanatory content."}[verdict]
        rows.append(_row("Pacing", f"Measured {cpm} visual cuts per minute; the usual range for this style is {lo:g}-{hi:g}. It is {verdict} range. {tip}",
                         "Measured from your file (scene-change detection, approximate).", "Medium"))
    else:
        rows.append(_row("Pacing", f"Aim for roughly {lo:g}-{hi:g} visual changes (cuts, zooms, B-roll, text) per minute.", "Style rule of thumb; upload the file for a measurement.", "Medium"))
    if media and media.get("available"):
        if media.get("loudness_lufs") is not None:
            L = media["loudness_lufs"]
            msg = ("Loudness is on target." if -16.5 <= L <= -12.5 else
                   "Too loud: YouTube will turn it down, and heavy limiting can sound crushed. Lower the master." if L > -12.5 else
                   "Quiet: YouTube does not raise quiet videos much. Raise voice and master toward about -14 LUFS.")
            rows.append(_row("Loudness", f"Integrated loudness {L:.1f} LUFS (target about -14, accept -16 to -13). {msg}", "Measured from your file (EBU R128).", "High" if abs(L + 14) > 4 else "Low"))
        if media.get("true_peak_dbfs") is not None and media["true_peak_dbfs"] > -1.0:
            rows.append(_row("Audio peaks", f"Peaks reach {media['true_peak_dbfs']:.1f} dBFS; keep peaks at or below -1 dBFS to avoid clipping after YouTube's encoding.", "Measured from your file.", "High"))
        if media.get("silent_gaps", 0) >= 3:
            rows.append(_row("Dead air", f"{media['silent_gaps']} silent gaps longer than 1.5 s were found; cut or cover them with music, B-roll or text.", "Measured from your file.", "Medium"))
        if is_short and not media.get("vertical"):
            rows.append(_row("Frame", f"The file is {media['width']}x{media['height']} (not vertical). Shorts work best at 9:16 (1080x1920).", "Measured from your file.", "High"))
        if not is_short and media.get("vertical"):
            rows.append(_row("Frame", "The file is vertical. For a standard long-form video use 16:9 at 1080p or higher, or publish it as a Short if it is 3 minutes or less.", "Measured from your file.", "Medium"))
        if media.get("height") and media["height"] < 720 and media.get("has_video"):
            rows.append(_row("Resolution", f"Only {media['height']}p; export at 1080p or higher if the source allows.", "Measured from your file.", "Medium"))
        if is_short and media.get("duration_seconds", 0) > 180:
            rows.append(_row("Shorts length", "Over 3 minutes will not be treated as a Short.", "Current Shorts length limit.", "High"))
    if is_short:
        rows.append(_row("Captions", "Burn in large, high-contrast captions inside the safe area (clear of the bottom UI); many people watch without sound.", "Common Shorts practice.", "Medium"))
    else:
        rows.append(_row("Structure", "Add chapters (timestamps in the description) and a pattern interrupt every 60-90 seconds: new angle, graphic, B-roll or a question.", "Common long-form practice.", "Medium"))
    eng = profile.get("engagement", {}) if profile.get("available") else {}
    cr = eng.get("short_comment_rate" if is_short else "long_comment_rate")
    if cr == cr and cr is not None:
        rows.append(_row("Audience prompt", f"Your public comment rate for this format is {cr * 100:.2f}% of views. End with one specific question viewers can answer in a few words.",
                         "Median comment rate of your videos in this format.", "Low"))
    return pd.DataFrame(rows)


def music_plan(fmt: str, mood: str, has_voice: bool = True) -> dict:
    spec = MOODS.get(mood, MOODS["Light & neutral"])
    cps = 60.0 / ((sum(spec["bpm"]) / 2) / spec["beats_per_cut"])
    return {"mood": mood, "genre": spec["genre"], "bpm": spec["bpm"], "cut_every_seconds": round(cps, 1), "beats_per_cut": spec["beats_per_cut"],
            "mix": (f"{spec['mix']} With a voice-over, keep the music about 12-18 dB below the voice and duck it a further 8-12 dB while speaking; "
                    "without speech, mix to about -14 LUFS overall." if has_voice else f"{spec['mix']} Mix to about -14 LUFS overall."),
            "structure": ("Start the music on the first frame, make the first beat land with the hook, and cut on beats." if fmt == "short" else
                          "Quiet bed during the intro and explanation, a lift at transitions and the payoff, and a clean outro of 5-10 seconds."),
            "sources": SOURCES,
            "ab_test": "Publish two similar videos with different music moods (e.g. calm vs upbeat). Compare views, likes per view and, in YouTube Studio, "
                       "average view duration and the retention graph. Keep what your own viewers respond to; one test is only a hint."}
