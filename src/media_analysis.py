"""Objective measurements of an uploaded video file using ffprobe/ffmpeg (if installed). The file is analysed locally and not stored.

Measures: duration, resolution, frame rate, audio loudness (LUFS) and true peak, silent gaps, cut rate, embedded music metadata.
It does NOT identify songs or detect copyright matches (that is YouTube Content ID, visible only in YouTube Studio).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mp3", ".wav", ".m4a"}
MAX_ANALYSE_SECONDS = 300
VIDEO_SCAN_SECONDS = 120  # cut-rate scan is the slowest step (it decodes video), so it covers the first 2 minutes only
TIMEOUT = 150


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffprobe") and shutil.which("ffmpeg"))


def _run(cmd: list[str], timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # list args, never a shell


def _fps(rate: str | None) -> float | None:
    try:
        return round(float(Fraction(rate)), 2) if rate and rate != "0/0" else None
    except (ValueError, ZeroDivisionError):
        return None


def analyze_media(path: str) -> dict:
    """Return a dict of measurements; any measurement that fails is None and explained in `notes`."""
    out: dict = {"available": False, "notes": []}
    if not ffmpeg_available():
        out["notes"].append("ffmpeg/ffprobe are not installed on this server, so the file could not be measured. Enter the details manually.")
        return out
    try:
        probe = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path], 60)
        info = json.loads(probe.stdout or "{}")
    except (subprocess.TimeoutExpired, ValueError, OSError):
        out["notes"].append("The file could not be read (corrupt or unsupported).")
        return out
    streams, fmt = info.get("streams", []), info.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v and not a:
        out["notes"].append("No audio or video stream found.")
        return out
    dur = float(fmt.get("duration") or (v or a).get("duration") or 0)
    w, h = (int(v.get("width", 0)), int(v.get("height", 0))) if v else (0, 0)
    rot = abs(int(float(((v or {}).get("tags") or {}).get("rotate", 0) or 0))) if v else 0
    if rot in (90, 270):
        w, h = h, w
    tags = {str(k).lower(): str(val) for k, val in {**(fmt.get("tags") or {}), **((a or {}).get("tags") or {})}.items()}
    out.update({"available": True, "duration_seconds": dur, "width": w, "height": h, "fps": _fps((v or {}).get("avg_frame_rate")),
                "vertical": bool(w and h and h > w), "has_video": bool(v), "has_audio": bool(a), "metadata_tags": tags,
                "analysed_seconds": min(dur, MAX_ANALYSE_SECONDS)})
    if dur > MAX_ANALYSE_SECONDS:
        out["notes"].append(f"Loudness, silence and cut rate were measured on the first {MAX_ANALYSE_SECONDS // 60} minutes only.")
    if a:
        try:
            r = _run(["ffmpeg", "-hide_banner", "-nostats", "-t", str(MAX_ANALYSE_SECONDS), "-i", path, "-vn", "-af",
                      "ebur128=peak=true,silencedetect=noise=-38dB:d=1.5", "-f", "null", "-"])
            log = r.stderr
            summary = log[log.rfind("Summary:"):] if "Summary:" in log else ""
            m = re.search(r"I:\s+(-?\d+\.?\d*)\s+LUFS", summary)
            p = re.search(r"Peak:\s+(-?\d+\.?\d*)\s+dBFS", summary)
            out["loudness_lufs"] = float(m.group(1)) if m else None
            out["true_peak_dbfs"] = float(p.group(1)) if p else None
            out["silent_gaps"] = len(re.findall(r"silence_start", log))
        except (subprocess.TimeoutExpired, OSError):
            out["notes"].append("Audio measurement timed out.")
    if v and dur > 0:
        try:
            r = _run(["ffmpeg", "-hide_banner", "-nostats", "-t", str(VIDEO_SCAN_SECONDS), "-i", path, "-an", "-vf",
                      "fps=4,scale=320:-2,select='gt(scene,0.30)',showinfo", "-f", "null", "-"])
            cuts = len(re.findall(r"pts_time:", r.stderr))
            out["cuts"] = cuts
            out["cuts_per_minute"] = round(cuts / max(min(dur, VIDEO_SCAN_SECONDS) / 60.0, 0.1), 1)
            if dur > VIDEO_SCAN_SECONDS:
                out["notes"].append(f"Cut rate was measured on the first {VIDEO_SCAN_SECONDS // 60} minutes only.")
        except (subprocess.TimeoutExpired, OSError):
            out["notes"].append("Cut-rate measurement timed out.")
    return out


def analyze_upload(data: bytes, filename: str) -> dict:
    """Write an uploaded file to a private temp path, measure it, and always delete it."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        return {"available": False, "notes": [f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXT))}."]}
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return analyze_media(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
