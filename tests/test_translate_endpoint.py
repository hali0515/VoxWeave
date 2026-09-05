"""Exercise translation through the real SDK without contacting an endpoint."""

import json
import sys
from types import SimpleNamespace

import httpx
import openai
import pytest
from click.testing import CliRunner

from voxweave import translate
from voxweave.cli import cli
from voxweave.progress import Reporter


BASE_URL = "https://translate.example.test/v1"
SYNTHETIC_KEY = "sk-synthetic-translation-test"
SERVED_MODEL = "Qwen3.8-27B-FP8"


def test_sdk3_stream_transport_errors_are_retryable_without_a_hard_dependency(
    monkeypatch,
):
    class TransportError(Exception):
        pass

    class ReadError(TransportError):
        pass

    monkeypatch.setitem(
        sys.modules, "httpx2", SimpleNamespace(TransportError=TransportError)
    )
    assert translate._retryable(ReadError("stream interrupted"))
    assert not translate._retryable(ValueError("invalid response"))


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    """Only the HTTP transport is replaced; SDK serialization stays real."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "VOXWEAVE_TRANSLATE_MODEL",
        "VOXWEAVE_TRANSLATE_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEST_TRANSLATE_KEY", SYNTHETIC_KEY)
    conf = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    requests = []
    clients = []
    sleeps = []
    real_openai = openai.OpenAI

    def install(handler):
        def record(request):
            requests.append(request)
            return handler(request)

        def make_client(**kwargs):
            # Isolate application retries from the SDK's independent retry loop.
            kwargs.setdefault("max_retries", 0)
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(record))
            client = real_openai(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(openai, "OpenAI", make_client)

    monkeypatch.setattr(translate, "_sleep", sleeps.append)
    monkeypatch.setattr(translate, "_RETRY_DELAYS", (0.0,))
    yield conf, install, requests, sleeps
    for client in clients:
        client.close()


def _config(path, *, effort=None, model=SERVED_MODEL, key_env="TEST_TRANSLATE_KEY"):
    values = {"model": model, "base_url": BASE_URL, "api_key_env": key_env}
    if effort is not None:
        values["reasoning_effort"] = effort
    path.write_text(
        "[llm]\n"
        + "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def _subtitle(tmp_path, count=1):
    path = tmp_path / "episode.vtt"
    path.write_text(
        "WEBVTT\n\n"
        + "\n\n".join(
            f"00:00:0{i + 1}.000 --> 00:00:0{i + 2}.000\nsource {i}"
            for i in range(count)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _body(request):
    return json.loads(request.content)


def _cue_ids(request):
    user = _body(request)["messages"][-1]["content"]
    return [cue["i"] for cue in json.loads(user)["cues"]]


def _models(model=SERVED_MODEL):
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "test"}
            ],
        },
    )


def _completion(request, translations):
    body = _body(request)
    content = json.dumps(
        {"translations": [{"i": i, "t": text} for i, text in translations.items()]}
    )
    common = {"id": "chatcmpl-test", "created": 0, "model": body["model"]}
    if not body.get("stream"):
        return httpx.Response(
            200,
            json={
                **common,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    # Reasoning deliberately contains plausible translation JSON. It must neither
    # advance cue progress nor become a subtitle or supersede the actual answer.
    deltas = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": '{"translations":[{"i":0,"t":"private reasoning"}]}',
        },
        {"content": content[: len(content) // 2]},
        {"content": content[len(content) // 2 :]},
    ]
    events = [
        {
            **common,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    events.append({**common, "object": "chat.completion.chunk", "choices": []})
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=stream + "data: [DONE]\n\n",
    )


def test_cli_auto_model_streams_with_exact_endpoint_key_and_effort(
    endpoint, tmp_path, monkeypatch
):
    conf, install, requests, _ = endpoint
    _config(conf, effort="low", model="auto")
    monkeypatch.setenv("VOXWEAVE_TRANSLATE_REASONING_EFFORT", "medium")
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"Ada": "Captain Ada"}', encoding="utf-8")
    install(
        lambda request: (
            _models()
            if request.method == "GET"
            else _completion(request, {0: "Welcome aboard"})
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "translate",
            str(_subtitle(tmp_path)),
            "--to",
            "en",
            "--reasoning-effort",
            "xhigh",
            "--context",
            "A spaceship reunion",
            "--glossary",
            str(glossary),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [(r.method, str(r.url)) for r in requests] == [
        ("GET", f"{BASE_URL}/models"),
        ("POST", f"{BASE_URL}/chat/completions"),
    ]
    assert all(
        r.headers["authorization"] == f"Bearer {SYNTHETIC_KEY}" for r in requests
    )
    body = _body(requests[-1])
    assert set(body) == {
        "model",
        "messages",
        "response_format",
        "stream",
        "reasoning_effort",
    }
    assert body["model"] == SERVED_MODEL
    assert body["reasoning_effort"] == "xhigh"
    assert body["stream"] is True
    assert body["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "A spaceship reunion" in body["messages"][0]["content"]
    assert "Captain Ada" in body["messages"][0]["content"]
    assert json.loads(body["messages"][1]["content"]) == {
        "cues": [{"i": 0, "t": "source 0"}]
    }
    output = (tmp_path / "episode.en.vtt").read_text(encoding="utf-8")
    assert "Welcome aboard" in output
    assert "private reasoning" not in output
    assert "00:00:01.000 --> 00:00:02.000" in output


@pytest.mark.parametrize(
    ("configured", "environment", "cli_effort", "expected"),
    [
        (None, None, None, None),
        ("low", None, None, "low"),
        ("low", "medium", None, "medium"),
        ("low", "medium", "default", None),
        ("low", "default", None, None),
    ],
    ids=["unset", "config", "environment", "cli-default", "environment-default"],
)
def test_cli_effort_precedence_and_explicit_endpoint_default(
    endpoint, tmp_path, monkeypatch, configured, environment, cli_effort, expected
):
    conf, install, requests, _ = endpoint
    _config(conf, effort=configured)
    if environment is not None:
        monkeypatch.setenv("VOXWEAVE_TRANSLATE_REASONING_EFFORT", environment)
    install(lambda request: _completion(request, {0: "translated"}))
    args = ["translate", str(_subtitle(tmp_path)), "--to", "en"]
    if cli_effort is not None:
        args += ["--reasoning-effort", cli_effort]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    body = _body(requests[0])
    if expected is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == expected
    assert "chat_template_kwargs" not in body
    assert "enable_thinking" not in body


@pytest.mark.parametrize("api_key", [None, "sk-explicit-synthetic"])
def test_library_uses_configured_endpoint_and_key_with_provider_effort(
    endpoint, api_key
):
    conf, install, requests, _ = endpoint
    _config(conf, effort="provider-budget-7")
    install(lambda request: _completion(request, {0: "translated"}))

    result = translate.translate_cues(
        [{"i": 0, "t": "source"}], to="en", model=SERVED_MODEL, api_key=api_key
    )

    assert result == {0: "translated"}
    assert len(requests) == 1
    assert str(requests[0].url) == f"{BASE_URL}/chat/completions"
    assert requests[0].headers["authorization"] == f"Bearer {api_key or SYNTHETIC_KEY}"
    assert _body(requests[0])["reasoning_effort"] == "provider-budget-7"
    assert not _body(requests[0]).get("stream", False)


def test_library_keyless_endpoint_and_explicit_default(endpoint):
    conf, install, requests, _ = endpoint
    _config(conf, effort="low", key_env="")
    install(lambda request: _completion(request, {0: "translated"}))

    result = translate.translate_cues(
        [{"i": 0, "t": "source"}],
        to="en",
        model=SERVED_MODEL,
        reasoning_effort="default",
    )

    assert result == {0: "translated"}
    assert str(requests[0].url) == f"{BASE_URL}/chat/completions"
    assert requests[0].headers["authorization"].startswith("Bearer ")
    assert requests[0].headers["authorization"] != "Bearer "
    assert "reasoning_effort" not in _body(requests[0])


@pytest.mark.parametrize("model", ["custom-subtitle-model", "auto"])
def test_library_omitted_model_uses_current_config_and_resolves_auto(endpoint, model):
    conf, install, requests, _ = endpoint
    _config(conf, model=model)
    install(
        lambda request: (
            _models()
            if request.method == "GET"
            else _completion(request, {0: "translated"})
        )
    )

    result = translate.translate_cues([{"i": 0, "t": "source"}], to="en")

    assert result == {0: "translated"}
    assert [request.method for request in requests] == (
        ["GET", "POST"] if model == "auto" else ["POST"]
    )
    assert _body(requests[-1])["model"] == (SERVED_MODEL if model == "auto" else model)
    if model == "auto":
        assert str(requests[0].url) == f"{BASE_URL}/models"


def test_cli_missing_cue_retry_preserves_effort(endpoint, tmp_path):
    conf, install, requests, _ = endpoint
    _config(conf, effort="xhigh")
    install(
        lambda request: _completion(
            request, {0: "first"} if len(requests) == 1 else {1: "second"}
        )
    )

    result = CliRunner().invoke(
        cli, ["translate", str(_subtitle(tmp_path, 2)), "--to", "en"]
    )

    assert result.exit_code == 0, result.output
    assert [_cue_ids(request) for request in requests] == [[0, 1], [1]]
    assert [_body(request)["reasoning_effort"] for request in requests] == [
        "xhigh",
        "xhigh",
    ]
    output = (tmp_path / "episode.en.vtt").read_text(encoding="utf-8")
    assert "first" in output and "second" in output


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_definitive_http_errors_fail_once_without_dropping_effort(endpoint, status):
    conf, install, requests, sleeps = endpoint
    _config(conf)
    install(
        lambda request: httpx.Response(
            status,
            json={
                "error": {
                    "message": "request rejected",
                    "type": "invalid_request_error",
                }
            },
        )
    )

    with pytest.raises(openai.APIStatusError) as failure:
        translate.translate_cues(
            [{"i": 0, "t": "source"}],
            to="en",
            model=SERVED_MODEL,
            reasoning_effort="xhigh",
        )

    assert failure.value.status_code == status
    assert len(requests) == 1
    assert _body(requests[0])["reasoning_effort"] == "xhigh"
    assert sleeps == []


@pytest.mark.parametrize("failure", [429, 503, "connection"])
def test_transient_retry_keeps_the_original_request(endpoint, failure):
    conf, install, requests, sleeps = endpoint
    _config(conf)

    def respond(request):
        if len(requests) == 1:
            if failure == "connection":
                raise httpx.ConnectError("synthetic connection loss", request=request)
            return httpx.Response(failure, json={"error": {"message": "try again"}})
        return _completion(request, {0: "translated"})

    install(respond)
    result = translate.translate_cues(
        [{"i": 0, "t": "source"}],
        to="en",
        model=SERVED_MODEL,
        reasoning_effort="medium",
    )

    assert result == {0: "translated"}
    assert len(requests) == 2
    assert _body(requests[0]) == _body(requests[1])
    assert _body(requests[1])["reasoning_effort"] == "medium"
    assert len(sleeps) == 1


@pytest.mark.parametrize("failure_type", [httpx.ReadError, httpx.ReadTimeout])
def test_interrupted_sdk_stream_retries_with_effort_and_closes_responses(
    endpoint, failure_type
):
    conf, install, requests, sleeps = endpoint
    _config(conf)
    streams = []
    advances = []

    class TrackedStream(httpx.SyncByteStream):
        def __init__(self, content, failure=None):
            self.content = content
            self.failure = failure
            self.closed = False

        def __iter__(self):
            yield self.content
            if self.failure is not None:
                raise self.failure

        def close(self):
            self.closed = True

    class RecordingReporter(Reporter):
        def advance(self, n=1):
            advances.append(n)

    def respond(request):
        if streams:
            assert streams[0].closed, "The failed response must close before retrying"
        content = _completion(request, {0: "complete translation"}).content
        failure = None
        if not streams:
            # Deliver reasoning plus half of the content before the socket fails.
            content = b"\n\n".join(content.split(b"\n\n")[:2]) + b"\n\n"
            failure = failure_type("synthetic midstream disconnect", request=request)
        stream = TrackedStream(content, failure)
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    install(respond)
    result = translate.translate_cues(
        [{"i": 0, "t": "source"}],
        to="en",
        model=SERVED_MODEL,
        reasoning_effort="xhigh",
        reporter=RecordingReporter(),
    )

    assert result == {0: "complete translation"}
    assert len(requests) == 2
    assert _body(requests[0]) == _body(requests[1])
    assert _body(requests[1])["reasoning_effort"] == "xhigh"
    assert _body(requests[1])["stream"] is True
    assert len(sleeps) == 1
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    # Each attempt sees one content entry; reasoning never advances the reporter.
    assert advances == [1, 1]


@pytest.mark.parametrize(
    "changed", [None, "effort", "served-model", "endpoint", "context", "glossary"]
)
def test_interrupted_cli_reuses_progress_only_for_the_same_request(
    endpoint, tmp_path, monkeypatch, changed
):
    conf, install, requests, _ = endpoint
    _config(conf, model="auto")
    subtitle = _subtitle(tmp_path, 2)
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"Ada": "Captain Ada"}', encoding="utf-8")
    # Make a tiny two-window episode while retaining real persistence and HTTP.
    monkeypatch.setattr(
        translate, "_plan_windows", lambda units, **kwargs: [[unit] for unit in units]
    )
    state = {"interrupted": True, "model": SERVED_MODEL}

    def respond(request):
        if request.method == "GET":
            return _models(state["model"])
        ids = _cue_ids(request)
        if state["interrupted"] and ids == [1]:
            raise KeyboardInterrupt
        prefix = "old" if state["interrupted"] else "fresh"
        return _completion(request, {i: f"{prefix} entry {i}" for i in ids})

    install(respond)
    args = [
        "translate",
        str(subtitle),
        "--to",
        "en",
        "--reasoning-effort",
        "low",
        "--context",
        "Original scene",
        "--glossary",
        str(glossary),
    ]
    first = CliRunner().invoke(cli, args)
    assert first.exit_code != 0
    assert [_cue_ids(r) for r in requests if r.method == "POST"] == [[0], [1]]
    assert list(tmp_path.rglob("*.progress.json"))

    state["interrupted"] = False
    if changed == "effort":
        args[args.index("--reasoning-effort") + 1] = "xhigh"
    elif changed == "served-model":
        state["model"] = "replacement-served-model"
    elif changed == "endpoint":
        args += ["--base-url", "https://replacement.example.test/v1"]
    elif changed == "context":
        args[args.index("--context") + 1] = "Changed scene"
    elif changed == "glossary":
        glossary.write_text('{"Ada": "Admiral Ada"}', encoding="utf-8")
    requests.clear()

    resumed = CliRunner().invoke(cli, args)

    assert resumed.exit_code == 0, resumed.output
    assert [_cue_ids(r) for r in requests if r.method == "POST"] == (
        [[1]] if changed is None else [[0], [1]]
    )
    output = (tmp_path / "episode.en.vtt").read_text(encoding="utf-8")
    assert "fresh entry 1" in output
    assert ("old entry 0" in output) is (changed is None)
    assert ("fresh entry 0" in output) is (changed is not None)
    assert not list(tmp_path.rglob("*.progress.json"))
