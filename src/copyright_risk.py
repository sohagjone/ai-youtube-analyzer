"""Copyright and reused/inauthentic-content RISK checker (a pre-upload screen, not a verdict).

What it CAN do: score risk from what the creator declares about music, clips, images and voice; scan title/description/tags for risky wording;
read embedded music metadata of an uploaded file; and compare the draft with the channel's own videos to spot template repetition.
What it CANNOT do: detect Content ID matches, identify songs or footage, or give legal advice. YouTube's own copyright check during upload in
YouTube Studio is the only check that sees Content ID. YouTube's monetization policy on 'inauthentic' (mass-produced/repetitive) and 'reused'
content is evaluated across the whole channel; commentary/transformation must add clear original value.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from utils.data_processing import term_set

MUSIC = {"None / only my own voice and sounds": 0, "YouTube Audio Library or Shorts sound library": 5, "Paid licence that covers monetized YouTube (I keep the licence)": 5,
         "I created or commissioned it (I hold the rights)": 0, "'No copyright'/royalty-free from a website (licence unclear)": 30,
         "Popular commercial song": 55, "I am not sure": 30}
CLIPS = {"None": 0, "Short clip (under 10 seconds) with my commentary or transformation": 15, "Several clips / a compilation": 40,
         "Large part of someone else's video (reupload or reaction without much added)": 65}
IMAGES = {"My own": 0, "Licensed stock (I keep the licence)": 3, "Creative Commons (with the required credit)": 8, "Found online (search results)": 30, "Brand logos / characters / celebrities": 30}
VOICE = {"My own voice or face": 0, "My own script read by a voice tool": 8, "A voice tool reading text I did not write": 35}
RISK_WORDS = [
    (r"\bno copyright( infringement)? intended\b", "Disclaimers like this do not give permission; they do not stop claims.", 8, "medium"),
    (r"\bfair use\b", "'Fair use' is a legal judgement made case by case; putting it in the description does not protect the video.", 8, "medium"),
    (r"\bcopyright[- ]free\b", "Check the actual licence of every track and clip; 'copyright free' labels are often wrong.", 12, "medium"),
    (r"\bfull (movie|episode|film|match|game|concert|album|season)\b", "Uploading complete films, episodes, matches or concerts is the classic reused/copyright pattern.", 30, "high"),
    (r"\b(official )?(lyrics?|lyric video|karaoke)\b", "Lyrics and lyric videos of songs you do not own usually need a licence.", 25, "high"),
    (r"\b(re-?upload|reupload|re-?post|mirror)\b", "Reuploads of other people's videos are reused content unless clearly transformed.", 30, "high"),
    (r"\b(highlights?|clips?|best moments)\b.*\b(nba|nfl|fifa|uefa|premier league|ufc|wwe|mlb|nhl|f1|netflix|disney|marvel|hbo)\b|\b(nba|nfl|fifa|uefa|premier league|ufc|wwe|mlb|nhl|f1|netflix|disney|marvel|hbo)\b.*\b(highlights?|clips?|best moments)\b",
     "Highlights of leagues and studio content are heavily policed by their rights owners.", 25, "high"),
    (r"\b(compilation|mashup|remix|nightcore|slowed\s*(\+|and)?\s*reverb|sped up)\b", "Compilations, remixes and speed-edited songs need rights to every source used and clear added value.", 18, "medium"),
    (r"\b(reaction|reacting to)\b", "Reactions are allowed when your commentary is substantial and your face/voice is clearly part of it; passive reactions are risky.", 8, "low"),
    (r"\b(download|mp3|torrent|crack(ed)?|leak(ed)?)\b", "Wording that suggests piracy or leaked material attracts enforcement.", 20, "high"),
]
SEV = {"high": 3, "medium": 2, "low": 1, "info": 0}


def _f(area: str, finding: str, why: str, fix: str, points: int, sev: str) -> dict:
    return {"area": area, "severity": sev, "finding": finding, "why_it_matters": why, "what_to_do": fix, "points": points}


def _self_similarity(title: str, description: str, own: pd.DataFrame | None) -> list[dict]:
    out: list[dict] = []
    if own is None or own.empty:
        return out
    near = [(r.title, SequenceMatcher(None, title.lower(), str(r.title).lower()).ratio()) for r in own.itertuples()]
    t, ratio = max(near, key=lambda x: x[1])
    if ratio >= 0.85:
        out.append(_f("Repetition", f"The title is {ratio:.0%} similar to your earlier video '{t}'.", "Near-identical videos look repetitive or mass-produced to reviewers and viewers.",
                      "Change the angle, add new information, or update the existing video instead.", 15, "medium"))
    ds = term_set(description)
    if len(ds) >= 15:
        best = max((len(ds & term_set(d)) / len(ds | term_set(d)) for d in own["description"].astype(str) if d.strip()), default=0.0)
        if best >= 0.8:
            out.append(_f("Repetition", f"The description overlaps {best:.0%} with an earlier one.", "Copy-paste metadata across many videos is a template signal.", "Write a description specific to this video.", 8, "low"))
    first3 = own.sort_values("published_at", ascending=False).head(15)["title"].astype(str).str.lower().str.split().str[:3].str.join(" ")
    if len(first3) >= 8:
        share = first3.value_counts().iloc[0] / len(first3)
        if share >= 0.6:
            out.append(_f("Repetition", f"{share:.0%} of your latest titles start with the same three words ('{first3.value_counts().index[0]}').",
                          "Mass-produced, templated uploads with minimal variation are what YouTube's 'inauthentic content' policy targets.",
                          "Vary the structure and make each video clearly different in substance.", 12, "medium"))
    return out


def assess(title: str, description: str = "", tags: list[str] | None = None, music: str = "I am not sure", clips: str = "None", images: str = "My own",
           voice: str = "My own voice or face", has_permission: bool = False, adds_commentary: bool = True, ai_realistic: bool = False, ai_disclosed: bool = False,
           own_df: pd.DataFrame | None = None, media: dict | None = None, duration_s: float | None = None) -> dict:
    text = f"{title}\n{description}\n{' '.join(tags or [])}".lower()
    rows: list[dict] = []
    mp = MUSIC.get(music, 30)
    if mp >= 30:
        rows.append(_f("Music", f"Music source: {music}.", "Content ID can claim, mute or block videos using music you cannot prove you may use; a claim can redirect all revenue.",
                       "Swap in a track from the YouTube Audio Library / Shorts sound library or a licence that covers monetized YouTube, and keep the licence.", mp, "high" if mp >= 50 else "medium"))
    elif mp:
        rows.append(_f("Music", f"Music source: {music}.", "Low risk, but terms still apply (attribution, monetization limits).", "Keep the licence or track page; add required credits.", mp, "low"))
    cp = CLIPS.get(clips, 0)
    if cp and has_permission:
        cp = int(cp * 0.25)
        rows.append(_f("Footage", f"Third-party footage ({clips}) with permission or licence.", "Permission must cover monetized YouTube use and be provable.", "Keep the written permission; credit as agreed.", cp, "low"))
    elif cp:
        sev = "high" if cp >= 40 else "medium"
        rows.append(_f("Footage", f"Third-party footage: {clips}.", "Rights owners claim clips automatically, and YouTube's reused-content policy requires clear original commentary, criticism or transformation.",
                       "Use less footage, add substantial commentary on screen, or get written permission/licence.", cp, sev))
        if not adds_commentary:
            rows.append(_f("Footage", "You said you add little original commentary or transformation.", "Without clear added value, reused footage is a monetization risk even where a copyright claim does not occur.",
                           "Add original explanation, analysis or a different creative structure that is clearly yours.", 25, "high"))
    ip = IMAGES.get(images, 0)
    if ip >= 8:
        rows.append(_f("Images", f"Images: {images}.", "Search-result images and brand/character/celebrity material often belong to someone else and can be claimed or disputed.", "Use your own, licensed stock, or properly credited Creative Commons media.", ip, "medium" if ip >= 30 else "low"))
    vp = VOICE.get(voice, 0)
    if vp >= 30:
        rows.append(_f("Voice / script", f"Voice: {voice}.", "Text taken from articles or other videos is reused content, and bulk synthetic narration over recycled material is what the inauthentic-content policy targets.",
                       "Write your own script with your own analysis, or get permission for the text.", vp, "high"))
    elif vp:
        rows.append(_f("Voice / script", f"Voice: {voice}.", "Fine when the script and visuals are yours; repetitive templated output across many videos is a risk.", "Keep each video distinct and add original footage or analysis.", vp, "low"))
    if ai_realistic and not ai_disclosed:
        rows.append(_f("AI disclosure", "The video has realistic AI-generated or altered content that is not disclosed.", "YouTube requires creators to disclose realistic synthetic or altered content in the upload settings.", "Turn on the altered/synthetic content disclosure in YouTube Studio.", 15, "medium"))
    for pat, why, pts, sev in RISK_WORDS:
        m = re.search(pat, text)
        if m:
            rows.append(_f("Wording", f"Found '{m.group(0).strip()}' in the title, description or tags.", why, "Reword if it does not describe your video; if it does, make sure you hold the rights.", pts, sev))
    rows += _self_similarity(title, description, own_df)
    if media and media.get("available"):
        meta = media.get("metadata_tags", {})
        hit = {k: v for k, v in meta.items() if k in {"artist", "album", "album_artist", "composer", "copyright", "publisher", "isrc", "label"} and v.strip()}
        if hit:
            rows.append(_f("File metadata", "The file carries music metadata: " + "; ".join(f"{k}={v[:40]}" for k, v in list(hit.items())[:4]) + ".",
                           "Embedded artist/album/copyright tags usually mean a commercial track is in the audio.", "Confirm you hold the rights to that audio or replace it.", 25, "high"))
    rows.append(_f("Check in Studio", "This tool cannot see Content ID.", "Only YouTube's own copyright check (run during upload) can show real matches.", "Wait for 'Checks' to finish before publishing; fix or trim flagged sections.", 0, "info"))
    f = pd.DataFrame(rows).sort_values(["points"], ascending=False).reset_index(drop=True)
    score = int(min(100, f["points"].sum()))
    level = "High" if score >= 50 else "Medium" if score >= 25 else "Low"
    return {"level": level, "score": score, "findings": f.drop(columns=["points"]),
            "summary": {"High": "Several things here are likely to attract claims or reused-content review. Fix the high-severity items before uploading.",
                        "Medium": "Some items could cause a claim or a monetization review. Address them or keep proof of your rights.",
                        "Low": "Nothing risky was declared or found in the text. This is not a guarantee: only YouTube's check sees Content ID matches."}[level],
            "disclaimer": "A pre-upload screen based on what you declared and what is visible in the text. It is not legal advice, cannot detect Content ID matches and does not predict "
                          "YouTube's decisions. When in doubt, use your own or properly licensed material."}
