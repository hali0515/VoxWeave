def test_ctc_observer_runs_on_flat_result_before_distribution(monkeypatch):
    from voxweave import align_ctc

    events = []
    flat = [
        {"text": "one", "start": 0.0, "end": 1.0},
        {"text": "two", "start": 1.0, "end": 2.0},
        {"hostile-surplus": object()},
    ]
    monkeypatch.setattr(
        align_ctc, "_ctc_build_tokens", lambda norm, nospace, al: ([1], [0], ["x"])
    )
    monkeypatch.setattr(align_ctc, "_ctc_emit_full", lambda al, wav: object())
    monkeypatch.setattr(align_ctc, "_ctc_align_logp", lambda *args: flat)
    original_distribute = align_ctc._distribute_units

    def distribute(units, texts, iso):
        events.append(("distribute", units))
        return original_distribute(units, texts, iso)

    monkeypatch.setattr(align_ctc, "_distribute_units", distribute)
    result = align_ctc._ctc_full_pass(
        object(),
        type("Wave", (), {"shape": [32_000]})(),
        ["one", "two"],
        False,
        "en",
        _raw_result_observer=lambda units: events.append(("observe", units)),
    )
    assert [event[0] for event in events] == ["observe", "distribute"]
    assert events[0][1] is flat
    assert result == [flat[:1], flat[1:2]]


def test_mms_observer_runs_on_flat_result_before_distribution(monkeypatch):
    from voxweave import align_mms

    events = []
    flat = [
        {"text": "a", "start": 0.0, "end": 1.0},
        {"text": "b", "start": 1.0, "end": 2.0},
    ]
    monkeypatch.setattr(align_mms, "_mms_emit_units", lambda wav, text, iso: flat)
    monkeypatch.setattr(align_mms, "_empty_cache", lambda: None)
    original_distribute = align_mms._distribute_units

    def distribute(units, texts, iso):
        events.append(("distribute", units))
        return original_distribute(units, texts, iso)

    monkeypatch.setattr(align_mms, "_distribute_units", distribute)
    result = align_mms._mms_full_pass(
        object(),
        ["a", "b"],
        "ja",
        _raw_result_observer=lambda units: events.append(("observe", units)),
    )
    assert [event[0] for event in events] == ["observe", "distribute"]
    assert events[0][1] is flat
    assert result == [flat[:1], flat[1:2]]
