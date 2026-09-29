"""Frontier-model generation. This module never chooses the next action."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from jevcode.util import clip


class GenerationError(Exception):
    pass


@dataclass(frozen=True)
class Profile:
    provider: str
    model: str
    api_key: str
    base_url: str | None


@dataclass
class UsageMeter:
    jev_calls: int = 0
    jev_input_tokens: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    def snapshot(self) -> "UsageMeter":
        return UsageMeter(
            self.jev_calls,
            self.jev_input_tokens,
            self.llm_calls,
            self.llm_input_tokens,
            self.llm_output_tokens,
        )

    def delta(self, earlier: "UsageMeter") -> "UsageMeter":
        return UsageMeter(
            self.jev_calls - earlier.jev_calls,
            self.jev_input_tokens - earlier.jev_input_tokens,
            self.llm_calls - earlier.llm_calls,
            self.llm_input_tokens - earlier.llm_input_tokens,
            self.llm_output_tokens - earlier.llm_output_tokens,
        )


@dataclass(frozen=True)
class LLMResult:
    text: str
    truncated: bool
    model: str


class LLM:
    def __init__(self, powerful: Profile, fast: Profile | None, meter: UsageMeter) -> None:
        self.powerful = powerful
        self.fast = fast or powerful
        self.meter = meter
        self._http = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0))

    def close(self) -> None:
        self._http.close()

    def complete(self, tier: str, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult:
        profile = self.fast if tier == "fast" else self.powerful
        try:
            if profile.provider == "anthropic":
                text, truncated = self._anthropic(profile, system, user, max_tokens, temperature)
            else:
                text, truncated = self._openai_compatible(profile, system, user, max_tokens, temperature)
        except httpx.HTTPError as error:
            raise GenerationError(f"{profile.provider} request failed: {error}") from error
        return LLMResult(text=text, truncated=truncated, model=profile.model)

    def _openai_compatible(
        self,
        profile: Profile,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, bool]:
        url = _chat_url(profile)
        headers = {"Content-Type": "application/json"}
        if profile.api_key:
            headers["Authorization"] = f"Bearer {profile.api_key}"
        token_key = "max_completion_tokens" if profile.provider == "openai" else "max_tokens"
        payload: dict = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            token_key: max_tokens,
        }
        data = self._post(url, headers, payload)
        self._add_usage(data)
        return _openai_text(data)

    def _anthropic(
        self,
        profile: Profile,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, bool]:
        headers = {
            "x-api-key": profile.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": profile.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._post("https://api.anthropic.com/v1/messages", headers, payload)
        self._add_usage(data)
        blocks = data.get("content") or []
        texts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not texts:
            raise GenerationError(f"empty anthropic response: {clip(str(data), 500)}")
        return "\n".join(texts), data.get("stop_reason") == "max_tokens"

    def _post(self, url: str, headers: dict[str, str], payload: dict) -> dict:
        body = dict(payload)
        response: httpx.Response | None = None
        last_error = ""
        for _ in range(3):
            response = self._http.post(url, headers=headers, json=body)
            if response.status_code < 400:
                return response.json()
            last_error = response.text[:1200]
            if response.status_code != 400 or not _repair_payload(body, last_error):
                break
        status = response.status_code if response is not None else 0
        raise GenerationError(f"generation failed ({status}): {last_error}")

    def _add_usage(self, data: dict) -> None:
        usage = data.get("usage") or {}
        incoming = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        outgoing = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        self.meter.llm_calls += 1
        self.meter.llm_input_tokens += int(incoming)
        self.meter.llm_output_tokens += int(outgoing)


def _repair_payload(body: dict, error_text: str) -> bool:
    lowered = error_text.lower()
    unrecognized = any(word in lowered for word in ("unknown", "unrecognized", "unsupported", "unexpected"))
    if (
        "max_completion_tokens" in body
        and "max_completion_tokens" in lowered
        and unrecognized
        and "max_tokens" not in lowered
    ):
        body["max_tokens"] = body.pop("max_completion_tokens")
        return True
    if "max_tokens" in body and "max_completion_tokens" in lowered and "max_tokens" in lowered:
        body["max_completion_tokens"] = body.pop("max_tokens")
        return True
    if "temperature" in body and "temperature" in lowered and unrecognized:
        body.pop("temperature", None)
        return True
    for key in ("max_tokens", "max_completion_tokens"):
        current = body.get(key)
        if isinstance(current, int) and current > 4096 and "maximum" in lowered:
            body[key] = 4096
            return True
    return False


def _chat_url(profile: Profile) -> str:
    base = (profile.base_url or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _openai_text(data: dict) -> tuple[str, bool]:
    choices = data.get("choices") or []
    if not choices:
        raise GenerationError(f"empty generation response: {clip(str(data), 500)}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("model returned no text")
    finish = str(choices[0].get("finish_reason") or "")
    return content, finish in {"length", "max_tokens"}
