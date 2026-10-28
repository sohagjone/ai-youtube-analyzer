"""Local analysis of an uploaded video file with a bundled ffmpeg (imageio-ffmpeg) - nothing leaves the server and
the temporary file is deleted by the caller. Measures facts (duration, size, loudness, silence, cuts, rough tempo);
it cannot judge quality, and cannot detect copyrighted material."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SCENE_THRESHOLD = 0.30
ANALYSIS_SECONDS = 180  # cut detection and tempo use at most the first 3 minutes to keep runs fast


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to a system install
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise FFmpegMissing("ffmpeg is not available (install the 'imageio-ffmpeg' package or ffmpeg).")


def _run(args: list[str], timeout: int = 240, binary: bool = False):
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", *args], capture_output=True, timeout=timeout,
                          text=not binary)


def probe(path: str | Path) -> dict:
    """Duration, resolution, fps, audio presence from `ffmpeg -i` output."""
    err = _run(["-i", str(path)]).stderr
    d = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    v = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
    fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    out = {
        "duration_seconds": int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else None,
        "width": int(v.group(1)) if v else None, "height": int(v.group(2)) if v else None,
        "fps": float(fps.group(1)) if fps else None, "has_audio": "Audio:" in err, "has_video": "Video:" in err,
    }
    if out["width"] and out["height"]:
        out["orientation"] = "vertical" if out["height"] > out["width"] else "horizontal" if out["width"] > out["height"] else "square"
        out["aspect_ratio"] = round(out["width"] / out["height"], 3)
    return out


def loudness(path: str | Path) -> dict:
    """Integrated loudness (LUFS), loudness range and true peak via ebur128. Empty dict when there is no audio."""
    err = _run(["-nostats", "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"]).stderr
    tail = err[err.rfind("Summary:"):] if "Summary:" in err else err
    i = re.search(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", tail)
    lra = re.search(r"LRA:\s+(\d+(?:\.\d+)?)\s+LU", tail)
    pk = re.search(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", tail)
    if not i:
        return {}
    return {"integrated_lufs": float(i.group(1)), "loudness_range_lu": float(lra.group(1)) if lra else None,
            "true_peak_dbfs": float(pk.group(1)) if pk else None}


def silence(path: str | Path, noise_db: int = -35, min_len: float = 0.7) -> dict:
    err = _run(["-i", str(path), "-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"]).stderr
    durations = [float(x) for x in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", err)]
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", err)]
    return {"silent_gaps": len(durations), "silent_seconds": round(sum(durations), 2),
            "starts_silent": bool(starts and starts[0] <= 0.3), "first_gap_start": starts[0] if starts else None}


def scene_cuts(path: str | Path, threshold: float = SCENE_THRESHOLD, seconds: int = ANALYSIS_SECONDS) -> list[float]:
    """Timestamps of visual cuts (scene-change score above threshold) in the first `seconds` seconds."""
    err = _run(["-t", str(seconds), "-i", str(path), "-an", "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]).stderr
    return [float(x) for x in re.findall(r"pts_time:(\d+(?:\.\d+)?)", err)]


def tempo_estimate(path: str | Path, seconds: int = 60) -> dict:
    """Very rough BPM from the onset autocorrelation of the audio. Only meaningful when music dominates the track."""
    res = _run(["-t", str(seconds), "-i", str(path), "-vn", "-ac", "1", "-ar", "11025", "-f", "s16le", "-"], binary=True)
    raw = res.stdout
    if len(raw) < 11025 * 2 * 8:
        return {}
    x = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
    frame, hop = 512, 256
    n = (len(x) - frame) // hop
    env = np.sqrt(np.array([np.mean(x[i * hop:i * hop + frame] ** 2) for i in range(n)]))
    onset = np.maximum(np.diff(env), 0)
    onset = onset - onset.mean()
    if not np.any(onset):
        return {}
    ac = np.correlate(onset, onset, "full")[len(onset) - 1:]
    fps = 11025 / hop
    lo, hi = int(60 * fps / 180), int(60 * fps / 60)
    seg = ac[lo:hi + 1]
    if len(seg) == 0 or ac[0] <= 0:
        return {}
    lag = lo + int(np.argmax(seg))
    bpm = 60 * fps / lag
    while bpm < 75:
        bpm *= 2
    while bpm > 150:
        bpm /= 2
    return {"bpm": round(float(bpm), 1), "confidence": round(float(ac[lag] / ac[0]), 3)}


def analyze_video(path: str | Path) -> dict:
    """All measurements in one dict. Raises FFmpegMissing, or ValueError for unreadable files."""
    info = probe(path)
    if not info.get("has_video") or not info.get("duration_seconds"):
        raise ValueError("This file could not be read as a video. Try an MP4 or MOV exported from your editor.")
    out = dict(info)
    dur = info["duration_seconds"]
    if info["has_audio"]:
        out["loudness"] = loudness(path)
        out["silence"] = silence(path)
        out["tempo"] = tempo_estimate(path)
    else:
        out["loudness"], out["silence"], out["tempo"] = {}, {}, {}
    cuts = scene_cuts(path)
    window = min(dur, ANALYSIS_SECONDS)
    out["cuts"] = {"count": len(cuts), "window_seconds": window, "cuts_per_minute": len(cuts) / window * 60 if window else 0.0,
                   "avg_shot_seconds": window / (len(cuts) + 1), "cuts_in_first_5s": sum(1 for c in cuts if c <= 5.0), "timestamps": cuts[:60]}
    return out


def analyze_bytes(data: bytes, suffix: str = ".mp4") -> dict:
    """Analyse an uploaded file from memory: writes a temporary file, analyses it, and ALWAYS deletes it."""
    import os
    import tempfile
    fd, name = tempfile.mkstemp(suffix=suffix if suffix.startswith(".") else "." + suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return analyze_video(name)
    finally:
        if os.path.exists(name):
            os.remove(name)
