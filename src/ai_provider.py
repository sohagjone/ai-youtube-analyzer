"""Optional AI provider (Anthropic or OpenAI-compatible) used via plain HTTPS. Core analysis never depends on it."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

import requests

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


class AIProviderError(Exception):
    """Raised for any AI problem; messages never contain credentials."""


class AIProvider:
    def __init__(self, provider: str = "anthropic", api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, session: requests.Session | None = None):
        self.provider = (provider or "anthropic").lower()
        if self.provider not in DEFAULT_MODELS:
            raise AIProviderError(f"Unsupported AI provider '{provider}'. Use 'anthropic' or 'openai'.")
        self._api_key = (api_key or "").strip()
        self.model = (model or "").strip() or DEFAULT_MODELS[self.provider]
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AIProvider":
        """Build from a mapping with AI_PROVIDER / AI_API_KEY / AI_MODEL (secrets or env)."""
        provider = str(config.get("AI_PROVIDER") or "anthropic")
        key = config.get("AI_API_KEY") or config.get(f"{provider.upper()}_API_KEY") or ""
        return cls(provider, str(key), str(config.get("AI_MODEL") or "") or None)

    @classmethod
    def from_env(cls) -> "AIProvider":
        return cls.from_config(os.environ)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def complete(self, system: str, user: str, max_tokens: int = 1500, images: list[bytes] | None = None) -> str:
        if not self.is_configured:
            raise AIProviderError("No AI API key configured.")
        import base64
        b64 = [base64.b64encode(i).decode() for i in (images or [])]
        try:
            if self.provider == "anthropic":
                content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}} for d in b64] + [{"type": "text", "text": user}]
                resp = self.session.post(
                    "https://api.anthropic.com/v1/messages", timeout=self.timeout,
                    headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": self.model, "max_tokens": max_tokens, "system": system,
                          "messages": [{"role": "user", "content": content if b64 else user}]})
            else:
                ocontent = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}} for d in b64] + [{"type": "text", "text": user}]
                resp = self.session.post(
                    "https://api.openai.com/v1/chat/completions", timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"},
                    json={"model": self.model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": ocontent if b64 else user}]})
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise AIProviderError(self._redact(f"Could not reach the AI provider ({type(exc).__name__}).")) from None
        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.json().get("error", {}).get("message", ""))
            except (ValueError, AttributeError):
                pass
            raise AIProviderError(self._redact(f"AI provider error {resp.status_code}: {detail[:200]}"))
        try:
            body = resp.json()
            if self.provider == "anthropic":
                return "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise AIProviderError("The AI provider returned an unexpected response.") from None

    def complete_json(self, system: str, user: str, max_tokens: int = 1500, images: list[bytes] | None = None) -> Any:
        return parse_json_loose(self.complete(system, user, max_tokens, images))


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in markdown fences or surrounded by prose."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        a, b = cleaned.find(open_c), cleaned.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(cleaned[a:b + 1])
            except ValueError:
                continue
    raise AIProviderError("The AI response was not valid JSON.")
