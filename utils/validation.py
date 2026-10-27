"""Input validation and channel-identifier parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

CHANNEL_ID_RE = re.compile(r"^UC[0-9A-Za-z_-]{22}$")
HANDLE_RE = re.compile(r"^@[A-Za-z0-9._\-]{3,30}$")
BARE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]{3,30}$")

VIDEO_PATH_HINTS = {"watch", "shorts", "playlist", "live", "embed", "results", "feed", "hashtag"}


@dataclass(frozen=True)
class ParsedChannel:
    """A normalised channel reference. kind is one of: id, handle, username, custom."""

    kind: str
    value: str

    @property
    def label(self) -> str:
        return self.value


def parse_channel_input(text: str | None) -> ParsedChannel:
    """Parse a channel URL, @handle or channel ID. Raises ValueError with a friendly message."""
    if text is None or not str(text).strip():
        raise ValueError("Enter a channel URL, @handle, or channel ID.")
    raw = str(text).strip()

    if CHANNEL_ID_RE.match(raw):
        return ParsedChannel("id", raw)

    if raw.startswith("@"):
        if HANDLE_RE.match(raw):
            return ParsedChannel("handle", raw)
        raise ValueError(f"'{raw}' is not a valid YouTube handle.")

    lowered = raw.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        candidate = raw if re.match(r"^https?://", raw, re.I) else "https://" + raw
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        if host == "youtu.be":
            raise ValueError("That looks like a video link. Paste a channel URL, @handle or channel ID.")
        if not (host == "youtube.com" or host.endswith(".youtube.com")):
            raise ValueError("Only youtube.com channel URLs are supported.")
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if not parts:
            raise ValueError("That YouTube URL does not point to a channel.")
        first = parts[0]
        if first.startswith("@"):
            if HANDLE_RE.match(first):
                return ParsedChannel("handle", first)
            raise ValueError(f"'{first}' is not a valid YouTube handle.")
        if first == "channel" and len(parts) > 1:
            if CHANNEL_ID_RE.match(parts[1]):
                return ParsedChannel("id", parts[1])
            raise ValueError("That channel ID is not valid (it should start with 'UC' and be 24 characters).")
        if first == "user" and len(parts) > 1:
            return ParsedChannel("username", parts[1])
        if first == "c" and len(parts) > 1:
            return ParsedChannel("custom", parts[1])
        if first in VIDEO_PATH_HINTS:
            raise ValueError("That looks like a video or playlist link. Paste a channel URL, @handle or channel ID.")
        raise ValueError("Unsupported YouTube URL format. Use a /@handle, /channel/UC..., /user/... or /c/... link.")

    if BARE_NAME_RE.match(raw):
        return ParsedChannel("handle", "@" + raw)
    raise ValueError(f"'{raw}' is not a recognisable channel URL, handle or ID.")


def parse_many(lines: list[str], limit: int = 5) -> tuple[list[ParsedChannel], list[str]]:
    """Parse competitor inputs. Returns (valid, error messages). Duplicates are dropped."""
    valid: list[ParsedChannel] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        if not line or not line.strip():
            continue
        try:
            parsed = parse_channel_input(line)
        except ValueError as exc:
            errors.append(f"{line.strip()}: {exc}")
            continue
        key = (parsed.kind, parsed.value.lower())
        if key in seen:
            continue
        seen.add(key)
        valid.append(parsed)
    if len(valid) > limit:
        errors.append(f"Only the first {limit} competitors are analysed; the rest were ignored.")
        valid = valid[:limit]
    return valid, errors
