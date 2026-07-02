"""LLM model chain with quota tracking, failover, and LA-midnight resets."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "google-genai is not installed in this Python environment. "
        "Run: venv\\Scripts\\pip install google-genai"
    ) from exc
from groq import APIStatusError, AsyncGroq, RateLimitError

from config import (
    GEMINI_TOKEN,
    GOOGLE_CHAT_MODEL,
    GOOGLE_VISION_MODEL,
    GROQ_TOKEN,
    MODEL_QUOTA_THRESHOLD,
    MODEL_USAGE_PATH,
    TIMEZONE,
)

NotifyFn = Callable[[str, str, str], Awaitable[None]]

_switch_notifier: NotifyFn | None = None


class Provider(str, Enum):
    GROQ = "groq"
    GOOGLE = "google"


@dataclass(frozen=True)
class ModelLimits:
    tpd: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    rpm: int | None = None


@dataclass(frozen=True)
class GoogleGenConfig:
    thinking_level: types.ThinkingLevel = types.ThinkingLevel.MINIMAL
    response_json: bool = True


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    provider: Provider
    limits: ModelLimits
    google_config: GoogleGenConfig | None = None


@dataclass
class CompletionResult:
    raw: str
    model_id: str
    display_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AllModelsExhausted(Exception):
    """Every model in the chain is rate-limited or over quota."""


# Groq free-tier limits (console.groq.com/settings/limits). TPD omitted where not listed.
GROQ_FREE_TIER_LIMITS: dict[str, ModelLimits] = {
    "groq/compound-mini": ModelLimits(tpm=70_000, rpd=250, rpm=30),
    "llama-3.1-8b-instant": ModelLimits(tpd=500_000, tpm=6_000, rpd=14_400, rpm=30),
    "llama-3.3-70b-versatile": ModelLimits(tpd=100_000, tpm=12_000, rpd=1_000, rpm=30),
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelLimits(
        tpd=500_000, tpm=30_000, rpd=1_000, rpm=30
    ),
    "meta-llama/llama-prompt-guard-2-22m": ModelLimits(
        tpd=500_000, tpm=15_000, rpd=14_400, rpm=30
    ),
    "meta-llama/llama-prompt-guard-2-86m": ModelLimits(
        tpd=500_000, tpm=15_000, rpd=14_400, rpm=30
    ),
    "openai/gpt-oss-120b": ModelLimits(tpd=200_000, tpm=8_000, rpd=1_000, rpm=30),
    "openai/gpt-oss-20b": ModelLimits(tpd=200_000, tpm=8_000, rpd=1_000, rpm=30),
    "openai/gpt-oss-safeguard-20b": ModelLimits(tpd=200_000, tpm=8_000, rpd=1_000, rpm=30),
    "qwen/qwen3-32b": ModelLimits(tpd=500_000, tpm=6_000, rpd=1_000, rpm=60),
    "qwen/qwen3.6-27b": ModelLimits(tpd=200_000, tpm=8_000, rpd=1_000, rpm=30),
}


def _groq_limits(model_id: str) -> ModelLimits:
    limits = GROQ_FREE_TIER_LIMITS.get(model_id)
    if limits is None:
        raise KeyError(f"No Groq free-tier limits configured for {model_id!r}")
    return limits


# Priority order (user spec). Groq limits from GROQ_FREE_TIER_LIMITS above.
MODEL_CHAIN: list[ModelSpec] = [
    ModelSpec(
        "openai/gpt-oss-120b",
        "GPT-OSS 120B",
        Provider.GROQ,
        _groq_limits("openai/gpt-oss-120b"),
    ),
    ModelSpec(
        GOOGLE_CHAT_MODEL,
        "Gemma (Google)",
        Provider.GOOGLE,
        ModelLimits(tpd=1_000_000, tpm=8_000, rpd=1_500, rpm=15),
        google_config=GoogleGenConfig(),
    ),
    ModelSpec(
        "llama-3.3-70b-versatile",
        "Llama 3.3 70B",
        Provider.GROQ,
        _groq_limits("llama-3.3-70b-versatile"),
    ),
    ModelSpec(
        "qwen/qwen3.6-27b",
        "Qwen 3.6 27B",
        Provider.GROQ,
        _groq_limits("qwen/qwen3.6-27b"),
    ),
    ModelSpec(
        "qwen/qwen3-32b",
        "Qwen 3 32B",
        Provider.GROQ,
        _groq_limits("qwen/qwen3-32b"),
    ),
    ModelSpec(
        "llama-3.1-8b-instant",
        "Llama 3.1 8B",
        Provider.GROQ,
        _groq_limits("llama-3.1-8b-instant"),
    ),
]

_groq_client = AsyncGroq(api_key=GROQ_TOKEN)
_google_client: genai.Client | None = None
_id_to_spec = {m.id: m for m in MODEL_CHAIN}

_GOOGLE_JSON_SUFFIX = (
    "\n\nYou are a JSON API. Respond with a single JSON object only. "
    "No markdown, no backticks, no reasoning, no preamble."
)


def _get_google_client() -> genai.Client:
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=GEMINI_TOKEN)
    return _google_client


def _is_gemma4_model(model_id: str) -> bool:
    return model_id.lower().startswith("gemma-4")


def _google_gen_config(spec: ModelSpec) -> GoogleGenConfig:
    if spec.google_config is not None:
        return spec.google_config
    if _is_gemma4_model(spec.id):
        return GoogleGenConfig()
    return GoogleGenConfig(response_json=True)


def _google_system_instruction(messages: list[dict]) -> str | None:
    parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    base = "\n\n".join(parts)
    combined = (base + _GOOGLE_JSON_SUFFIX).strip()
    return combined or None


def _google_contents(messages: list[dict]) -> list[types.Content]:
    contents: list[types.Content] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        content = msg.get("content", "")
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)],
            )
        )
    return contents


def _google_answer_text(response) -> str:
    """Return model answer text, skipping Gemma thought-channel parts."""
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def set_switch_notifier(fn: NotifyFn | None) -> None:
    global _switch_notifier
    _switch_notifier = fn


def _groq_model_kwargs(model_id: str) -> dict:
    m = model_id.lower()
    extra: dict = {}
    if "qwen" in m:
        extra["reasoning_effort"] = "none"
    elif "gpt-oss" in m:
        extra["reasoning_effort"] = "low"
    if not extra:
        return {}
    return {"extra_body": extra}


def _la_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def _la_date_key(now: datetime | None = None) -> str:
    return (_la_now() if now is None else now).date().isoformat()


def _minute_key(now: datetime | None = None) -> str:
    t = _la_now() if now is None else now
    return t.strftime("%Y-%m-%dT%H:%M")


def _parse_limit_reason(err_text: str) -> str:
    text = (err_text or "").lower()
    if "tokens per day" in text or "tpd" in text:
        return "daily token"
    if "tokens per minute" in text or "tpm" in text:
        return "tokens per minute"
    if "requests per day" in text or "rpd" in text:
        return "daily request"
    if "requests per minute" in text or "rpm" in text:
        return "requests per minute"
    if "rate_limit" in text or "429" in text:
        return "rate"
    return "rate"


_RETRY_AFTER_RE = re.compile(
    r"try again in (?P<secs>\d+)m(?P<extra>\d+(?:\.\d+)?)?s",
    re.I,
)


def _parse_retry_after_seconds(err_text: str) -> int | None:
    m = _RETRY_AFTER_RE.search(err_text or "")
    if not m:
        return None
    mins = int(m.group("secs"))
    extra = m.group("extra")
    secs = int(float(extra)) if extra else 0
    return mins * 60 + secs


@dataclass
class _ModelUsage:
    tokens: int = 0
    requests: int = 0


@dataclass
class UsageLedger:
    la_date: str = ""
    minute_key: str = ""
    active_model_id: str = ""
    usage_day: dict[str, _ModelUsage] = field(default_factory=dict)
    usage_minute: dict[str, _ModelUsage] = field(default_factory=dict)
    exhausted_day: set[str] = field(default_factory=set)
    exhausted_until: dict[str, str] = field(default_factory=dict)  # model_id -> iso datetime

    @classmethod
    def load(cls, path: Path) -> UsageLedger:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        ledger = cls(
            la_date=data.get("la_date", ""),
            minute_key=data.get("minute_key", ""),
            active_model_id=data.get("active_model_id", ""),
            exhausted_day=set(data.get("exhausted_day", [])),
            exhausted_until=dict(data.get("exhausted_until", {})),
        )
        for mid, u in (data.get("usage_day") or {}).items():
            ledger.usage_day[mid] = _ModelUsage(
                tokens=int(u.get("tokens", 0)),
                requests=int(u.get("requests", 0)),
            )
        for mid, u in (data.get("usage_minute") or {}).items():
            ledger.usage_minute[mid] = _ModelUsage(
                tokens=int(u.get("tokens", 0)),
                requests=int(u.get("requests", 0)),
            )
        return ledger

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "la_date": self.la_date,
            "minute_key": self.minute_key,
            "active_model_id": self.active_model_id,
            "usage_day": {
                k: {"tokens": v.tokens, "requests": v.requests}
                for k, v in self.usage_day.items()
            },
            "usage_minute": {
                k: {"tokens": v.tokens, "requests": v.requests}
                for k, v in self.usage_minute.items()
            },
            "exhausted_day": sorted(self.exhausted_day),
            "exhausted_until": self.exhausted_until,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def rollover_if_needed(self, now: datetime | None = None) -> None:
        now = now or _la_now()
        today = _la_date_key(now)
        minute = _minute_key(now)
        if self.la_date != today:
            self.la_date = today
            self.usage_day.clear()
            self.exhausted_day.clear()
            self.exhausted_until.clear()
        if self.minute_key != minute:
            self.minute_key = minute
            self.usage_minute.clear()

    def _day(self, model_id: str) -> _ModelUsage:
        if model_id not in self.usage_day:
            self.usage_day[model_id] = _ModelUsage()
        return self.usage_day[model_id]

    def _minute(self, model_id: str) -> _ModelUsage:
        if model_id not in self.usage_minute:
            self.usage_minute[model_id] = _ModelUsage()
        return self.usage_minute[model_id]

    def record(self, model_id: str, prompt: int, completion: int) -> None:
        total = prompt + completion
        d = self._day(model_id)
        d.tokens += total
        d.requests += 1
        m = self._minute(model_id)
        m.tokens += total
        m.requests += 1

    def _at_threshold(self, used: int, limit: int | None) -> bool:
        if not limit:
            return False
        return used >= int(limit * MODEL_QUOTA_THRESHOLD)

    def is_exhausted(self, spec: ModelSpec, now: datetime | None = None) -> bool:
        now = now or _la_now()
        if spec.id in self.exhausted_day:
            return True
        until_raw = self.exhausted_until.get(spec.id)
        if until_raw:
            try:
                until = datetime.fromisoformat(until_raw)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=ZoneInfo(TIMEZONE))
                if now < until:
                    return True
                del self.exhausted_until[spec.id]
            except ValueError:
                del self.exhausted_until[spec.id]

        lim = spec.limits
        day = self._day(spec.id)
        minute = self._minute(spec.id)
        if self._at_threshold(day.tokens, lim.tpd):
            return True
        if self._at_threshold(day.requests, lim.rpd):
            return True
        if self._at_threshold(minute.tokens, lim.tpm):
            return True
        if self._at_threshold(minute.requests, lim.rpm):
            return True
        return False

    def mark_exhausted(
        self,
        spec: ModelSpec,
        reason: str,
        *,
        now: datetime | None = None,
        retry_after_sec: int | None = None,
    ) -> None:
        now = now or _la_now()
        if reason in ("daily token", "daily request"):
            self.exhausted_day.add(spec.id)
            return
        if retry_after_sec and retry_after_sec > 0:
            until = now + timedelta(seconds=retry_after_sec + 1)
        elif reason in ("tokens per minute", "requests per minute"):
            until = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        else:
            until = now + timedelta(minutes=5)
        self.exhausted_until[spec.id] = until.isoformat()

    def check_proactive_thresholds(self, spec: ModelSpec) -> str | None:
        """Mark exhausted at 95% and return reason if crossed."""
        lim = spec.limits
        day = self._day(spec.id)
        minute = self._minute(spec.id)
        if self._at_threshold(day.tokens, lim.tpd):
            self.exhausted_day.add(spec.id)
            return "daily token"
        if self._at_threshold(day.requests, lim.rpd):
            self.exhausted_day.add(spec.id)
            return "daily request"
        if self._at_threshold(minute.tokens, lim.tpm):
            self.mark_exhausted(spec, "tokens per minute")
            return "tokens per minute"
        if self._at_threshold(minute.requests, lim.rpm):
            self.mark_exhausted(spec, "requests per minute")
            return "requests per minute"
        return None


_ledger = UsageLedger.load(Path(MODEL_USAGE_PATH))


def _chain_start_index() -> int:
    if _ledger.active_model_id:
        for i, spec in enumerate(MODEL_CHAIN):
            if spec.id == _ledger.active_model_id:
                return i
    return 0


async def _notify_switch(from_spec: ModelSpec | None, to_spec: ModelSpec, reason: str) -> None:
    if _switch_notifier is None:
        return
    from_name = from_spec.display_name if from_spec else "primary"
    try:
        await _switch_notifier(from_name, to_spec.display_name, reason)
    except Exception as e:
        print(f"⚠️ model switch notifier failed: {e}")


async def _groq_complete(
    spec: ModelSpec,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int, int]:
    response = await _groq_client.chat.completions.create(
        model=spec.id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **_groq_model_kwargs(spec.id),
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    return raw, prompt, completion, total


async def _google_complete(
    spec: ModelSpec,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int, int]:
    gcfg = _google_gen_config(spec)
    config_kwargs: dict = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    system_instruction = _google_system_instruction(messages)
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if gcfg.response_json:
        config_kwargs["response_mime_type"] = "application/json"
    if _is_gemma4_model(spec.id):
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=gcfg.thinking_level,
        )
    config = types.GenerateContentConfig(**config_kwargs)
    contents = _google_contents(messages)
    client = _get_google_client()

    def _run():
        return client.models.generate_content(
            model=spec.id,
            contents=contents,
            config=config,
        )

    response = await asyncio.to_thread(_run)
    raw = _google_answer_text(response)
    meta = getattr(response, "usage_metadata", None)
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    completion = int(getattr(meta, "candidates_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0) or (prompt + completion)
    return raw, prompt, completion, total


async def _complete_one(
    spec: ModelSpec,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int, int]:
    if spec.provider == Provider.GROQ:
        return await _groq_complete(
            spec, messages, max_tokens=max_tokens, temperature=temperature
        )
    if spec.provider == Provider.GOOGLE:
        return await _google_complete(
            spec, messages, max_tokens=max_tokens, temperature=temperature
        )
    raise ValueError(f"Unknown provider: {spec.provider}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIStatusError)):
        if getattr(exc, "status_code", None) == 429:
            return True
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "rate limit" in text


async def complete_chat(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.3,
) -> CompletionResult:
    """Try models in priority order; failover on quota / 429."""
    global _ledger
    now = _la_now()
    _ledger.rollover_if_needed(now)
    path = Path(MODEL_USAGE_PATH)

    start = _chain_start_index()
    ordered = MODEL_CHAIN[start:] + MODEL_CHAIN[:start]

    last_error = ""
    tried_any = False
    failed_spec: ModelSpec | None = None
    fail_reason = "rate"
    switch_notified = False

    for spec in ordered:
        if _ledger.is_exhausted(spec, now):
            continue

        if (
            not switch_notified
            and failed_spec is None
            and _ledger.active_model_id
            and spec.id != _ledger.active_model_id
        ):
            old = _id_to_spec.get(_ledger.active_model_id)
            if old and _ledger.is_exhausted(old, now):
                await _notify_switch(old, spec, "quota")
                switch_notified = True

        tried_any = True
        try:
            raw, prompt, completion, total = await _complete_one(
                spec,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            err_text = str(e)
            last_error = err_text
            fail_reason = _parse_limit_reason(err_text)
            retry_sec = _parse_retry_after_seconds(err_text)
            _ledger.mark_exhausted(
                spec, fail_reason, now=now, retry_after_sec=retry_sec
            )
            _ledger.save(path)
            failed_spec = spec
            print(
                f"⚠️ {spec.display_name} rate limited ({fail_reason}), trying next model"
            )
            continue

        if failed_spec is not None and not switch_notified:
            await _notify_switch(failed_spec, spec, fail_reason)
            switch_notified = True

        _ledger.record(spec.id, prompt, completion)
        if _ledger.active_model_id != spec.id:
            _ledger.active_model_id = spec.id

        proactive = _ledger.check_proactive_thresholds(spec)
        if proactive:
            print(f"⚠️ {spec.display_name} crossed 95% {proactive} quota")

        _ledger.save(path)
        return CompletionResult(
            raw=raw,
            model_id=spec.id,
            display_name=spec.display_name,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    if not tried_any:
        raise AllModelsExhausted(
            "All models are at capacity right now — try again in a few minutes."
        )
    retry_hint = _parse_retry_after_seconds(last_error)
    if retry_hint:
        mins = retry_hint // 60
        secs = retry_hint % 60
        raise AllModelsExhausted(
            f"All models are rate-limited — try again in about {mins}m {secs}s."
        )
    raise AllModelsExhausted(
        "All models are rate-limited right now — try again shortly."
    )


def _vision_spec(model_id: str | None = None) -> ModelSpec:
    mid = model_id or GOOGLE_VISION_MODEL
    return ModelSpec(
        mid,
        "Gemma Vision",
        Provider.GOOGLE,
        ModelLimits(tpd=1_000_000, tpm=8_000, rpd=1_500, rpm=15),
        google_config=GoogleGenConfig(response_json=False),
    )


async def complete_google_vision(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    *,
    model_id: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> CompletionResult:
    """Google-only vision completion with quota tracking (no Groq failover)."""
    global _ledger
    spec = _vision_spec(model_id)
    now = _la_now()
    _ledger.rollover_if_needed(now)
    path = Path(MODEL_USAGE_PATH)

    if _ledger.is_exhausted(spec, now):
        raise AllModelsExhausted(
            "Vision model is at capacity right now — try again in a few minutes."
        )

    config_kwargs: dict = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if _is_gemma4_model(spec.id):
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
        )
    config = types.GenerateContentConfig(**config_kwargs)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ],
        )
    ]
    client = _get_google_client()

    def _run():
        return client.models.generate_content(
            model=spec.id,
            contents=contents,
            config=config,
        )

    try:
        response = await asyncio.to_thread(_run)
    except Exception as e:
        if not _is_rate_limit_error(e):
            raise
        err_text = str(e)
        fail_reason = _parse_limit_reason(err_text)
        retry_sec = _parse_retry_after_seconds(err_text)
        _ledger.mark_exhausted(
            spec, fail_reason, now=now, retry_after_sec=retry_sec
        )
        _ledger.save(path)
        raise AllModelsExhausted(
            "Vision model is rate-limited right now — try again shortly."
        ) from e

    raw = _google_answer_text(response)
    meta = getattr(response, "usage_metadata", None)
    prompt_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
    completion_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0) or (
        prompt_tok + completion_tok
    )

    _ledger.record(spec.id, prompt_tok, completion_tok)
    proactive = _ledger.check_proactive_thresholds(spec)
    if proactive:
        print(f"⚠️ {spec.display_name} crossed 95% {proactive} quota")
    _ledger.save(path)

    return CompletionResult(
        raw=raw,
        model_id=spec.id,
        display_name=spec.display_name,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        total_tokens=total,
    )
