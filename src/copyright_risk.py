"""Copyright and reused-content RISK SCREEN.

This is a checklist-and-text screen, NOT a copyright detector. It cannot see YouTube's Content ID database, cannot recognise
music or footage, and is not legal advice. It flags common risk signals from what you declare and from your draft text, and
checks the draft against your own earlier titles and other channels' titles for near-duplicates. The authoritative checks are
YouTube Studio's own 'Checks' on your private upload (copyright and ad suitability) and the rights holders' decisions.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

DISCLAIMER = ("Risk screen only. It cannot detect Content ID matches or recognise copyrighted music/footage, and it is not legal advice. "
              "Before publishing, upload as Private/Unlisted and read YouTube Studio's 'Checks' results. Fair use is decided case by case; "
              "a disclaimer such as 'no copyright intended' does not grant permission.")

RISK_PHRASES = {
    "full movie": 20, "full episode": 20, "full match": 20, "full album": 20, "official video": 12, "official audio": 12,
    "lyrics": 12, "lyric video": 12, "ft.": 5, "feat.": 5, "remix": 8, "cover": 5, "reupload": 20, "re-upload": 20,
    "match highlights": 12, "live stream of": 8, "download free": 10, "free mp3": 15, "ripped": 15, "leaked": 15, "trailer": 5,
    "full song": 15, "tv show": 8, "movie clip": 12,
}
MYTH_PHRASES = ["no copyright intended", "fair use", "all rights belong", "i do not own", "i don't own", "for educational purposes only", "credit to the owner"]
MUSIC = {"none": 0, "own": 0, "licensed": 0, "unlicensed": 35, "unsure": 15}
FOOTAGE = {"all_original": 0, "licensed": 0, "others_clips": 30, "unsure": 10}


def _row(sev: str, issue: str, why: str, action: str, pts: int) -> dict:
    return {"severity": sev, "issue": issue, "why_it_matters": why, "what_to_do": action, "points": pts}


def screen(title: str, description: str, tags: list[str] | None, declared: dict, own_titles: list[str] | None = None,
           reference_titles: list[str] | None = None, repeated_structure_share: float | None = None) -> dict:
    """declared keys: music, footage, lyrics_or_scripts (bool), commentary (bool), ai_generated (bool), templated (bool), reupload (bool)."""
    text = f"{title} {description} {' '.join(tags or [])}".lower()
    rows: list[dict] = []
    mu, fo = declared.get("music", "unsure"), declared.get("footage", "unsure")
    if mu == "unlicensed":
        rows.append(_row("High", "Music without a licence", "Unlicensed music commonly triggers Content ID claims, muted audio, blocked videos or strikes.",
                         "Replace with music you own or hold a licence for (YouTube Audio Library or a paid library) and keep the licence proof.", MUSIC[mu]))
    elif mu == "unsure":
        rows.append(_row("Medium", "Music licence unknown", "You have not confirmed where the music comes from.", "Confirm the source and licence before publishing; credit it in the description.", MUSIC[mu]))
    if fo == "others_clips":
        pts = 18 if declared.get("commentary") else FOOTAGE[fo]
        rows.append(_row("High" if pts >= 25 else "Medium", "Clips or footage from other creators, TV, films or sports",
                         "Third-party footage is a common source of claims; commentary or criticism can help but does not guarantee fair use.",
                         "Use the minimum needed, add substantial original commentary/transformation, or obtain permission. Prefer licensed or original footage.", pts))
    elif fo == "unsure":
        rows.append(_row("Medium", "Footage source unknown", "You have not confirmed who owns all the footage and images.", "List the source and permission for every clip, image and sound.", FOOTAGE[fo]))
    if declared.get("lyrics_or_scripts"):
        rows.append(_row("Medium", "Lyrics, scripts or large text from other works", "Reproducing text from songs, books or shows can infringe.", "Quote only short excerpts for commentary, or write your own.", 15))
    if declared.get("reupload"):
        rows.append(_row("High", "Re-upload of your own earlier video (or near-identical content)", "Repetitive uploads can be treated as low-value and hurt monetisation eligibility.",
                         "Publish only if you add clear new value (updated information, new footage); otherwise update the original.", 25))
    if declared.get("ai_generated") and declared.get("templated"):
        rows.append(_row("High", "Mass-produced, templated, AI-generated content", "YouTube's monetisation policy targets mass-produced or repetitive content with little original input.",
                         "Add your own voice, analysis or footage, and vary the structure between videos.", 20))
    elif declared.get("templated") or (repeated_structure_share is not None and repeated_structure_share >= 0.6):
        rows.append(_row("Low", "Highly repetitive format", "Very similar videos can look mass-produced.", "Vary angles, titles and structure; show original input in each video.", 8))
    hits = [(p, w) for p, w in RISK_PHRASES.items() if p in text]
    if hits:
        pts = min(25, sum(w for _, w in hits))
        rows.append(_row("Medium" if pts >= 12 else "Low", "Rights-sensitive wording: " + ", ".join(p for p, _ in hits[:5]),
                         "Titles and descriptions that advertise full films, songs, matches or re-uploads attract claims and reviews.",
                         "Rename to describe your own original contribution (review, analysis, tutorial) and avoid implying you are publishing the original work.", pts))
    myths = [m for m in MYTH_PHRASES if m in text]
    if myths:
        rows.append(_row("Low", "Disclaimer wording that does not protect you: " + ", ".join(myths), "Statements like these do not give permission or establish fair use.",
                         "Rely on permission, licences or genuine transformation instead of disclaimers.", 5))
    low = title.strip().lower()
    dup = max(((SequenceMatcher(None, low, t.lower()).ratio(), t) for t in (own_titles or [])), default=(0.0, ""))
    if dup[0] >= 0.85:
        rows.append(_row("Medium", f"Title is {dup[0]:.0%} similar to your earlier video: '{dup[1]}'", "Near-duplicate videos look repetitive to viewers and to review systems.",
                         "Differentiate the title and the content, or update the earlier video instead.", 15))
    near = max(((SequenceMatcher(None, low, t.lower()).ratio(), t) for t in (reference_titles or [])), default=(0.0, ""))
    if near[0] >= 0.9:
        rows.append(_row("Low", f"Title almost identical to another channel's video: '{near[1]}'", "Copying another creator's title can confuse viewers and look like reuse.",
                         "Write your own angle; do not copy titles or thumbnails.", 10))
    score = int(min(100, sum(r["points"] for r in rows)))
    level = "Low" if score < 25 else "Medium" if score < 55 else "High"
    findings = pd.DataFrame(rows, columns=["severity", "issue", "why_it_matters", "what_to_do", "points"])
    if not findings.empty:
        order = {"High": 0, "Medium": 1, "Low": 2}
        findings = findings.sort_values("severity", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    checklist = ["Own or licensed music, with the licence saved", "Own or licensed footage, images and fonts", "Substantial original commentary, analysis or footage",
                 "Title/description don't imply you are publishing someone else's work", "Upload Private first and read YouTube Studio 'Checks'", "Keep proof of permissions"]
    return {"risk_score": score, "level": level, "findings": findings, "checklist": checklist, "disclaimer": DISCLAIMER,
            "summary": "No risk signals found in what you declared and wrote - this is not a clearance." if findings.empty else f"{len(findings)} risk signal(s) found."}
