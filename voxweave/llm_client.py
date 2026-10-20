"""LLM transport shared by :mod:`voxweave.translate` and :mod:`voxweave.asrfix`.

Client construction, the single chat call (blocking or streamed), finish-reason
checks, JSON salvage, the retry ladder and the json_object fallback. Nothing here
knows about cues, windows or glossaries.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

from voxweave import config

log = logging.getLogger("voxweave")

# Injectable for tests (backoff sleeps must not slow the suite down).
_sleep = time.sleep
# Two backoffs = three attempts per window; long enough to ride out rate-limit
# blips, short enough that a hard failure surfaces within seconds.
_RETRY_DELAYS = (2.0, 8.0)

# Per-request timeout (seconds) for the OpenAI client. The SDK's own retries are
# disabled (our _RETRY_DELAYS loop already retries), so a hung endpoint fails after
# at most three timeouts instead of the SDK's 600 s x 3 under each of ours.
LLM_TIMEOUT_S = config._env_float("VOXWEAVE_LLM_TIMEOUT_S", 300.0)
# Characters of the response kept in an IncompleteResponse message for diagnosis.
_INCOMPLETE_TAIL_CHARS = 120

_T = TypeVar("_T")


class IncompleteResponse(RuntimeError):
    """The model's answer cannot be used as a window result.

    Raised when the response (streamed or not) ended with a ``finish_reason``
    other than ``"stop"`` -- a length cap, a server-side abort (vLLM kills a
    request whose structured-output FSM fails to advance), or a stream that
    ended without any finish chunk -- or when the returned text carried no
    parseable translation object at all. Retryable: the retry loop re-requests
    the window and finally drops ``json_object`` mode (see
    :func:`_request_window`).
    """

    def __init__(
        self, finish_reason: str | None, text: str, *, detail: str | None = None
    ) -> None:
        self.finish_reason = finish_reason
        self.chars = len(text)
        self.tail = text[-_INCOMPLETE_TAIL_CHARS:]
        what = detail or f"finish_reason={finish_reason!r}"
        super().__init__(
            f"model response incomplete ({what}; {self.chars} chars received, "
            f"tail={self.tail!r})"
        )


def _loads_salvage(raw: object) -> dict:
    """Decode a model response to a dict, salvaging the first complete JSON object from dirty
    text (markdown fences / prose preamble). Returns {} when nothing parseable is found.

    Shared by parse_response (translations) and asrfix.parse_fixes (fixes).
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}  # non-str/non-dict (e.g. None): tolerate like the original inline parsers
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try every "{" in turn, not just the first: vLLM's guided decoding on
        # Qwen3.8 emits a stray '{"' before the real object, and a prose preamble
        # may contain braces of its own.
        decoder = json.JSONDecoder()
        obj = None
        start = raw.find("{")
        while start >= 0:
            try:
                obj, _ = decoder.raw_decode(raw[start:])
                break
            except json.JSONDecodeError:
                start = raw.find("{", start + 1)
        if obj is None:
            return {}
    return obj if isinstance(obj, dict) else {}


def _make_client(base_url: str | None, api_key: str | None):
    """Lazily import openai and construct a client; raises a clear error if the package is missing."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "translate requires openai (not installed); reinstall voxweave "
            "(openai is a core dependency)"
        ) from e
    base_url = config.resolve_llm_base_url(base_url)
    key_env = config.resolve_llm_api_key_env(None)
    key = api_key or (
        config.KEYLESS_API_KEY if key_env == "" else os.environ.get(key_env)
    )
    if not key:
        raise RuntimeError(
            f"{key_env} is not set; pass api_key= or set the {key_env} environment "
            'variable ([llm].api_key_env = "" declares a keyless endpoint)'
        )
    if base_url:
        return OpenAI(
            api_key=key, base_url=base_url, max_retries=0, timeout=LLM_TIMEOUT_S
        )
    return OpenAI(api_key=key, max_retries=0, timeout=LLM_TIMEOUT_S)


def resolve_model(client, model: str) -> str:
    """Turn ``"auto"`` into the endpoint's only served model; pass anything else through.

    Self-hosted servers (vLLM) change their served name whenever the operator
    swaps the weights; pinning that name in config turns every swap into a silent
    404. ``auto`` asks ``GET /v1/models`` at run time instead and refuses to guess
    when the endpoint serves more than one model.
    """
    if model != config.LLM_MODEL_AUTO:
        return model
    ids = [m.id for m in client.models.list()]
    if len(ids) == 1:
        log.info("llm model auto -> %s", ids[0])
        return ids[0]
    if not ids:
        raise RuntimeError("model 'auto': the endpoint serves no models")
    raise RuntimeError(
        f"model 'auto': the endpoint serves {len(ids)} models ({', '.join(ids)}); "
        "pick one with --model or [llm].model"
    )


def _check_complete(finish_reason: object, text: str) -> None:
    """Raise :class:`IncompleteResponse` unless the response finished with ``"stop"``.

    ``None`` means no finish reason was reported: a stream that ended without its
    finish chunk (the server dropped the request mid-answer), or a response that
    left the field out (the SDK exposes a missing field as ``None``, and so does
    :func:`_call` for a client object without the attribute). ``"length"`` /
    ``"abort"`` / anything else means the text is cut short -- a truncated JSON
    object never parses, so treating the window as "no result" and retrying beats
    silently accepting a partial buffer.
    """
    if finish_reason == "stop":
        return
    raise IncompleteResponse(
        finish_reason if isinstance(finish_reason, str) else None, text
    )


def _call(
    client,
    model: str,
    messages: list[dict],
    on_entry=None,
    *,
    reasoning_effort: str | None = None,
    json_mode: bool = True,
) -> str:
    """Call the model and return the JSON text.

    With ``on_entry``: streams and counts ``"i"`` keys as they arrive (one per cue),
    calling ``on_entry(delta)`` to advance the progress bar. Counting ``"i"`` is reliable
    because index values are numbers and translated strings JSON-escape interior quotes to
    ``\\"`` — no bare ``"i"`` tokens leak through. Without ``on_entry``: single blocking call.

    ``json_mode=False`` drops ``response_format={"type": "json_object"}`` (plain chat;
    the prompt already demands JSON and :func:`_loads_salvage` copes with fences and
    prose) -- the fallback for servers whose structured-output path aborts requests.

    Both paths check the final ``finish_reason`` and raise :class:`IncompleteResponse`
    unless it is ``"stop"`` (see :func:`_check_complete`).
    """
    request_options: dict = (
        {"reasoning_effort": reasoning_effort}
        if reasoning_effort is not None and reasoning_effort != "default"
        else {}
    )
    if json_mode:
        request_options["response_format"] = {"type": "json_object"}
    if on_entry is None:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            **request_options,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        _check_complete(getattr(choice, "finish_reason", None), text)
        return text

    buf: list[str] = []
    carry = ""
    finish_reason: object = None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **request_options,
    )
    try:
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue  # trailing usage-only chunks have no choices
            choice = chunk.choices[0]
            reason = getattr(choice, "finish_reason", None)
            if reason is not None:
                # A reported reason is final: a later None (servers may trail an
                # empty-delta chunk) must not erase a "stop" already seen.
                finish_reason = reason
            piece = getattr(getattr(choice, "delta", None), "content", None)
            if not piece:
                continue
            buf.append(piece)
            # Count only what arrived, plus the 2 characters a '"i"' split across
            # chunks can start in; rescanning the whole buffer is quadratic.
            window = carry + piece
            n = window.count('"i"')
            carry = window[-2:]
            if n:
                on_entry(n)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    text = "".join(buf)
    _check_complete(finish_reason, text)
    return text


def _retryable(exc: Exception) -> bool:
    """Retry transport failures, transient HTTP statuses and incomplete answers,
    never bad requests."""
    if isinstance(exc, IncompleteResponse):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 429} or 500 <= status < 600
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    # SSE body errors escape unwrapped: SDK 2 uses httpx, SDK 3 uses httpx2.
    for package in ("httpx", "httpx2"):
        try:
            transport_error = importlib.import_module(package).TransportError
        except ImportError:
            continue
        if isinstance(exc, transport_error):
            return True
    try:
        from openai import APIConnectionError
    except ImportError:
        return False
    return isinstance(exc, APIConnectionError)


def _with_retry(fn: Callable[[], _T], *, label: str = "translate call") -> _T:
    """Run ``fn`` with exponential backoff on transient failures (network, rate
    limit, 5xx, incomplete answers); non-retryable errors and the last attempt's
    error propagate."""
    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            return fn()
        except Exception as e:
            if delay is None or not _retryable(e):
                raise
            log.warning(
                "%s failed (attempt %d/%d: %s); retrying in %.0fs",
                label,
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                e,
                delay,
            )
            _sleep(delay)
    raise AssertionError("unreachable")


def request_with_json_fallback(
    attempt: Callable[[bool], _T],
    *,
    label: str,
    call_label: str | None = None,
) -> _T:
    """Run ``attempt(json_mode=True)`` with retries, then ``attempt(json_mode=False)``
    once when every retried attempt ended :class:`IncompleteResponse`.

    vLLM's structured-output grammar can abort a request (speculative decoding +
    reasoning parser); dropping ``response_format`` on the final attempt gets the
    request through on such servers. The final plain-chat failure propagates -- the
    caller decides what an unusable answer means for it.

    Shared by :func:`_request_window` (translate) and :mod:`voxweave.asrfix`; both
    request paths must behave the same in front of a flaky endpoint.
    """
    try:
        return _with_retry(lambda: attempt(True), label=call_label or label)
    except IncompleteResponse as exc:
        log.warning(
            "%s: json_object mode dropped after %d incomplete attempts (%s); "
            "retrying once as plain chat -- check the server's structured-output "
            "path (vLLM: guided decoding + speculative decoding / reasoning parser)",
            label,
            len(_RETRY_DELAYS) + 1,
            exc,
        )
    return attempt(False)
