"""Frontier-model generation. This module never chooses the next action."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from rift.util import clip


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


# Jev bills input only. Writer rates are USD per million tokens, input then output.
# Standard short-context public prices: OpenAI gpt-6-astra $10 / $50, gpt-6-luna $0.10 / $0.50;
# Anthropic claude-sonnet-5-5 $2 / $10. grok-4 follows the current grok-4 family rate $2 / $6.
JEV_USD_PER_MILLION_INPUT = 0.042
_MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6-luna": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "grok-4": (2.0, 6.0),
}
_PROVIDER_RATES: dict[str, tuple[float, float]] = {
    "openai": _MODEL_RATES["gpt-6-astra"],
    "anthropic": _MODEL_RATES["claude-sonnet-5-5"],
    "grok": _MODEL_RATES["grok-4"],
    "ollama": (0.0, 0.0),
}


def writer_rates(provider: str, model: str) -> tuple[float, float]:
    """Input and output USD per million tokens for the selected writer."""
    if model in _MODEL_RATES:
        return _MODEL_RATES[model]
    return _PROVIDER_RATES.get(provider, (0.0, 0.0))


def session_costs(meter: UsageMeter, provider: str, model: str) -> tuple[float, float]:
    """Session totals for Jev and the writer. The meter is not reset between tasks."""
    jev: float = meter.jev_input_tokens / 1_000_000 * JEV_USD_PER_MILLION_INPUT
    input_rate, output_rate = writer_rates(provider, model)
    writer: float = meter.llm_input_tokens / 1_000_000 * input_rate + meter.llm_output_tokens / 1_000_000 * output_rate
    return jev, writer


@dataclass(frozen=True)
class LLMResult:
    text: str
    truncated: bool
    model: str


class LLM:
    def __init__(self, powerful: Profile, fast: Profile | None, meter: UsageMeter, effort: str = "medium") -> None:
        self.powerful: Profile = powerful
        self.fast: Profile = fast or powerful
        self.meter: UsageMeter = meter
        self.effort: str = effort
        self._http = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0))

    def close(self) -> None:
        self._http.close()

    def complete(
        self,
        tier: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        effort: str | None = None,
    ) -> LLMResult:
        profile: Profile = self.fast if tier == "fast" else self.powerful
        chosen = self.effort if effort is None else effort
        try:
            if profile.provider == "anthropic":
                text, truncated = self._anthropic(profile, system, user, max_tokens, temperature, chosen)
            else:
                text, truncated = self._openai_compatible(
                    profile, system, user, max_tokens, temperature, chosen
                )
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
        effort: str,
    ) -> tuple[str, bool]:
        url: str = _chat_url(profile)
        headers: dict[str, str] = {"Content-Type": "application/json"}
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
        if profile.provider == "openai" and effort not in {"", "off"}:
            payload["reasoning_effort"] = effort
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
        effort: str,
    ) -> tuple[str, bool]:
        headers: dict[str, str] = {
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
        if effort not in {"", "off"}:
            payload["output_config"] = {"effort": effort}
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
    if "reasoning_effort" in body and "reasoning_effort" in lowered and unrecognized:
        body.pop("reasoning_effort", None)
        return True
    if "output_config" in body and "effort" in lowered and unrecognized:
        body.pop("output_config", None)
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


# Hidden reasoning is not a reply. A short budget often returns only these parts.
_REASONING_PARTS = {
    "reasoning",
    "reasoning_text",
    "reasoning_content",
    "thinking",
    "redacted_reasoning",
    "redacted_thinking",
    "summary_text",
}


def _openai_text(data: dict) -> tuple[str, bool]:
    """Visible reply text, ignoring reasoning traces.

    Reasoning models put the answer in content parts, output_text, or refusal
    while message.content is null or holds only a thinking block.
    """
    if not isinstance(data, dict):
        raise GenerationError("model returned no text")
    choices = data.get("choices") or []
    finish = ""
    text = ""
    if choices and isinstance(choices[0], dict):
        choice = choices[0]
        finish = str(choice.get("finish_reason") or "")
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = _visible_message_text(message)
        if not text.strip():
            legacy = choice.get("text")
            if isinstance(legacy, str):
                text = legacy
    if not text.strip():
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            text = output_text
    if not text.strip():
        text = _output_list_text(data.get("output"))
    if not text.strip():
        if not choices:
            raise GenerationError(f"empty generation response: {clip(str(data), 500)}")
        raise GenerationError("model returned no text")
    return text, finish in {"length", "max_tokens"}


def _visible_message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, (list, dict)):
        found = _visible_parts(content if isinstance(content, list) else [content])
        if found.strip():
            return found
    for key in ("output_text", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return ""


def _visible_parts(items: list) -> str:
    answers: list[str] = []
    refusals: list[str] = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                answers.append(item)
            continue
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").lower()
        if kind in _REASONING_PARTS:
            continue
        if kind == "refusal":
            found = _part_field(item, ("refusal", "text", "content"))
            if found.strip():
                refusals.append(found)
            continue
        found = _part_field(item, ("text", "output_text", "content", "refusal"))
        if found.strip():
            answers.append(found)
    if answers:
        return "".join(answers)
    return "".join(refusals)


def _part_field(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            nested = _visible_parts(value)
            if nested.strip():
                return nested
        if isinstance(value, dict):
            nested = _visible_parts([value])
            if nested.strip():
                return nested
    return ""


def _output_list_text(output: object) -> str:
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").lower()
        if kind in _REASONING_PARTS:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content)
            continue
        if isinstance(content, (list, dict)):
            parts = content if isinstance(content, list) else [content]
            found = _visible_parts(parts)
            if found.strip():
                chunks.append(found)
                continue
        found = _part_field(item, ("text", "output_text", "refusal"))
        if found.strip():
            chunks.append(found)
    return "\n".join(chunks)
