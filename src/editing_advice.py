"""Video-editing and music advice from (a) measurements of the uploaded file and (b) your channel's public performance.

Two kinds of evidence are labelled on every item: 'Measured in your file', 'From your channel data' and 'Rule of thumb'
(common editing guidance, not a law). No advice is a promise of better performance, and no track is recommended by name
- choose only music you own or hold a licence for.
"""
from __future__ import annotations

import pandas as pd

# niche -> (mood, bpm range, example genres, notes). Rules of thumb, not measurements.
MUSIC_GUIDE = {
    "gaming": ("energetic, driving", (115, 140), ["electronic", "synthwave", "drum & bass (lighter)"], "keep it under the game audio and your voice"),
    "sports": ("high-energy, building", (120, 150), ["hip-hop instrumental", "rock/electronic", "orchestral hybrid"], "match cuts to beats for highlight moments"),
    "tech": ("clean, modern, light", (95, 125), ["minimal electronic", "lo-fi tech", "corporate-light"], "stay low so spoken details are clear"),
    "education": ("calm, focused, unobtrusive", (70, 100), ["lo-fi", "ambient", "light acoustic"], "music should never compete with explanations"),
    "finance": ("steady, confident, minimal", (80, 110), ["corporate-light", "minimal piano", "ambient electronic"], "avoid anything that feels like hype"),
    "food": ("warm, upbeat", (90, 120), ["acoustic", "jazzy lo-fi", "light funk"], "match the mood of the dish or setting"),
    "travel": ("uplifting, open", (95, 125), ["indie/folk", "cinematic", "chill electronic"], "let big visuals breathe, cut on beats"),
    "fitness": ("driving, motivating", (125, 150), ["electronic", "hip-hop", "pop instrumental"], "tempo near the workout pace works well"),
    "music": ("n/a - the music is the content", (60, 180), ["your own recordings"], "check licences for any cover or sample"),
    "beauty": ("soft, stylish", (90, 115), ["chill pop", "soft electronic", "lo-fi"], "keep levels gentle under voice-over"),
    "comedy": ("playful, punchy", (100, 130), ["quirky instrumentals", "funk", "sound-effect led"], "short stingers and silence can serve jokes"),
    "general": ("neutral, warm", (85, 120), ["lo-fi", "light acoustic", "soft electronic"], "pick a mood that fits the first 10 seconds"),
}
LUFS_TARGET = -14.0


def _item(area: str, finding: str, advice: str, basis: str, priority: str = "Medium") -> dict:
    return {"area": area, "finding": finding, "advice": advice, "evidence": basis, "priority": priority}


def music_recommendation(ctx: dict, is_short: bool, detected: dict | None = None) -> dict:
    niche = ctx.get("niche", "general")
    mood, (lo, hi), genres, note = MUSIC_GUIDE.get(niche, MUSIC_GUIDE["general"])
    if is_short:
        lo, hi = min(hi, lo + 10), min(160, hi + 10)  # shorts usually run slightly faster
    rec = {"niche": niche, "mood": mood, "bpm_low": lo, "bpm_high": hi, "genres": genres, "note": note,
           "voice_music_mix": "If there is speech, keep music roughly 15-20 dB below the voice (about -30 to -35 LUFS under -14 LUFS speech); raise it in speech-free moments.",
           "licence": "Use music you made, or tracks you hold a licence for (e.g. YouTube Audio Library or a paid library). Keep the licence text/link and credit it in the description.",
           "basis": "Rule of thumb for your detected niche; refine it with your own results over several uploads."}
    if detected and detected.get("bpm") and detected.get("confidence", 0) >= 0.25:
        b = detected["bpm"]
        rec["detected_bpm"] = b
        rec["fit"] = "inside" if lo <= b <= hi else "outside"
    return rec


def advise(probe: dict | None, ctx: dict, fmt: str, music_status: str = "not decided") -> pd.DataFrame:
    """Return a table of prioritised advice items."""
    is_short = fmt == "Short"
    items: list[dict] = []
    dur = probe.get("duration_seconds") if probe else None
    # ---- structure from channel data
    if ctx.get("available"):
        if ctx.get("best_duration_bucket"):
            items.append(_item("Length", f"Your best-pacing length bucket (median views/day, older videos) is {ctx['best_duration_bucket']}." + (f" This video is {dur / 60:.1f} min." if dur else ""),
                               "If the story allows, aim near that bucket; cut anything that doesn't move the story forward.", "From your channel data", "Medium"))
        if ctx.get("preferred_format") and ((is_short and ctx["preferred_format"].startswith("Long")) or (not is_short and ctx["preferred_format"].startswith("Short"))):
            items.append(_item("Format", f"Your {ctx['preferred_format'].split(' (')[0].lower()} videos currently gain views faster than this format.",
                               "Consider testing this idea as the format that has worked better for you, or run it as a deliberate experiment.", "From your channel data", "Medium"))
    if not probe:
        items.append(_item("Upload a file", "No video file was analysed.", "Upload the exported video to get measured feedback on loudness, silence, cuts and hook.", "n/a", "Low"))
        return pd.DataFrame(items)
    # ---- format and technical
    w, h = probe.get("width"), probe.get("height")
    if is_short and probe.get("orientation") != "vertical":
        items.append(_item("Framing", f"File is {probe.get('orientation')} ({w}x{h}).", "Shorts are vertical (9:16). Re-frame or re-export vertically.", "Measured in your file", "High"))
    if not is_short and probe.get("orientation") == "vertical":
        items.append(_item("Framing", f"File is vertical ({w}x{h}) but planned as long-form.", "Long-form is normally 16:9; vertical long videos work but look small on TV/desktop.", "Measured in your file", "Medium"))
    if h and min(w or 0, h) < 1080 and (w or 0) * (h or 0) < 1920 * 1080 * 0.9:
        items.append(_item("Resolution", f"Resolution is {w}x{h}.", "Export at 1080p or higher if your source footage allows.", "Measured in your file", "Low"))
    if is_short and dur and dur > 180:
        items.append(_item("Length", f"Video is {dur:.0f}s; Shorts must be 3 minutes or less.", "Trim to 3 minutes or less, or publish as long-form.", "Measured in your file", "High"))
    # ---- hook and pacing
    cuts = probe.get("cuts", {})
    asl = cuts.get("avg_shot_seconds")
    if asl:
        lo, hi = (1.5, 4.0) if is_short else (3.0, 10.0)
        if asl > hi:
            items.append(_item("Pacing", f"Average shot length is {asl:.1f}s (rule of thumb for {'Shorts' if is_short else 'long-form'}: {lo}-{hi}s).",
                               "Add cutaways, zooms or b-roll to refresh the picture more often - only where it serves the content.", "Measured in your file + rule of thumb", "Medium"))
        elif asl < lo:
            items.append(_item("Pacing", f"Average shot length is {asl:.1f}s - very fast.", "Check that key moments stay on screen long enough to read or absorb.", "Measured in your file + rule of thumb", "Low"))
    if cuts.get("cuts_in_first_5s", 0) == 0 and dur and dur > 8:
        items.append(_item("Hook", "No visual change in the first 5 seconds.", "Open with your strongest moment, a clear promise or movement; avoid long intros and logos.", "Measured in your file + rule of thumb", "High"))
    sil = probe.get("silence", {})
    if sil.get("starts_silent"):
        items.append(_item("Hook", "The audio starts silent.", "Start sound (voice, music or effect) in the first second.", "Measured in your file", "High"))
    if sil.get("silent_gaps", 0) >= 3 and dur:
        items.append(_item("Pacing", f"{sil['silent_gaps']} silent gaps of 0.7s+ ({sil['silent_seconds']:.1f}s in total).", "Trim dead air unless the pause is deliberate.", "Measured in your file", "Medium"))
    # ---- audio
    if not probe.get("has_audio"):
        items.append(_item("Audio", "The file has no audio track.", "Add voice, music or effects - silent videos rarely hold attention.", "Measured in your file", "High"))
    else:
        ld = probe.get("loudness", {})
        if ld.get("integrated_lufs") is not None:
            l = ld["integrated_lufs"]
            if l < LUFS_TARGET - 4:
                items.append(_item("Loudness", f"Integrated loudness is {l:.1f} LUFS - quiet.", f"Raise to about {LUFS_TARGET:.0f} LUFS (YouTube's normalisation reference); quiet videos feel weak next to others.", "Measured in your file", "High"))
            elif l > LUFS_TARGET + 3:
                items.append(_item("Loudness", f"Integrated loudness is {l:.1f} LUFS - loud.", f"Reduce toward {LUFS_TARGET:.0f} LUFS; YouTube turns louder audio down, so extra loudness is lost and can add distortion.", "Measured in your file", "Medium"))
            else:
                items.append(_item("Loudness", f"Integrated loudness is {l:.1f} LUFS - near the {LUFS_TARGET:.0f} LUFS reference.", "Keep it.", "Measured in your file", "Low"))
            pk = ld.get("true_peak_dbfs")
            if pk is not None and pk > -1.0:
                items.append(_item("Loudness", f"True peak is {pk:.1f} dBFS.", "Use a limiter at about -1 dBTP to avoid clipping after YouTube's re-encoding.", "Measured in your file", "Medium"))
    # ---- music
    rec = music_recommendation(ctx, is_short, probe.get("tempo"))
    msg = f"Recommended mood for your {rec['niche']} niche: {rec['mood']}, about {rec['bpm_low']}-{rec['bpm_high']} BPM ({', '.join(rec['genres'])})."
    if "fit" in rec:
        msg += f" Your track's rough tempo is {rec['detected_bpm']:.0f} BPM ({rec['fit']} that range)."
    items.append(_item("Music", msg, rec["note"].capitalize() + ". " + rec["voice_music_mix"], "Rule of thumb for your niche" + (" + measured tempo" if "fit" in rec else ""), "Medium"))
    if music_status in ("licensed", "unlicensed", "not decided"):
        items.append(_item("Music licence", {"licensed": "You say the music is licensed.", "unlicensed": "You say the music is NOT licensed.", "not decided": "Music not decided yet."}[music_status],
                           rec["licence"], "Your declaration", "High" if music_status == "unlicensed" else "Low"))
    order = {"High": 0, "Medium": 1, "Low": 2}
    df = pd.DataFrame(items)
    return df.sort_values("priority", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
