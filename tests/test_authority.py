# tests/test_authority.py
"""Sealed issuance: one finalizer root per materialized row (P5 spec section 2.6, N8b).

RED skeleton for W1: ``voxweave.core.authority`` does not exist yet.

The gate this module exists for is not "did the right type reach ``finalize``" --
an ``isinstance`` check is satisfied by anything a test can build. It is "did the
stream this row delivered descend from THAT row's own upstream authority, and did
nothing else even try". So authority is issuer identity plus a sealed payload
digest, the capability is single-use, and an UNUSED laundering attempt is itself a
failure: a recorded event nobody consumed still says someone minted a root the
matrix did not expect.
"""

import pytest


def authority():
    from voxweave.core import authority as module

    return module


def finalizer():
    from voxweave.core import finalizer as module

    return module


def cue(text="a", start=0.0, end=1.0):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"text": text, "start": start, "end": end}],
        "speech_start": start,
        "speech_end": end,
    }


def profile(language="en", **over):
    from voxweave.core.segdoc import DisplayProfile

    base = dict(
        language=language,
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=0.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11.0 / 24.0,
    )
    base.update(over)
    return DisplayProfile(**base)


# --------------------------------------------------------------- primitives


def test_ledger_issues_a_verifiable_seal():
    module = authority()
    ledger = module.AuthorityLedger()
    payload = {"cues": [["a", 0.0, 1.0]]}
    capability = ledger.issue(issuer="test", kind="v1-capture", payload=payload)
    assert capability.seal.kind == "v1-capture"
    assert capability.seal.digest == module.digest_payload(payload)
    ledger.verify(capability.seal, payload)  # does not raise


def test_hand_built_seal_is_rejected_as_unissued():
    """R8-1(a): a STRUCTURALLY VALID hand-built authority is not an authority."""
    module = authority()
    ledger = module.AuthorityLedger()
    payload = {"cues": []}
    forged = module.Seal(
        issuer="test",
        authority_id="a0",
        kind="v1-capture",
        digest=module.digest_payload(payload),
    )
    with pytest.raises(module.UnissuedAuthority):
        ledger.verify(forged, payload)


def test_post_issuance_mutation_breaks_the_digest():
    """R8-1(b): the seal covers the payload, so a later edit cannot ride along."""
    module = authority()
    ledger = module.AuthorityLedger()
    payload = {"cues": [["a", 0.0, 1.0]]}
    capability = ledger.issue(issuer="test", kind="v1-capture", payload=payload)
    payload["cues"].append(["b", 1.0, 2.0])
    with pytest.raises(module.SealBroken):
        ledger.verify(capability.seal, payload)


def test_capability_is_single_use():
    module = authority()
    ledger = module.AuthorityLedger()
    payload = {"cues": []}
    capability = ledger.issue(issuer="test", kind="v1-capture", payload=payload)
    capability.consume(payload)
    with pytest.raises(module.CapabilityConsumed):
        capability.consume(payload)


def test_authority_kinds_are_closed_and_sorted():
    module = authority()
    assert module.AUTHORITY_KINDS == (
        "fresh-alignment",
        "optimizer-selection",
        "v1-capture",
    )


# --------------------------------------------------------------- root probe


def event(module, **over):
    base = dict(
        evaluation_id="e0",
        row_id="delivery_finalizer/v1",
        call_id="c0",
        input_seed_id="s0",
        input_kind="phase1",
        parent_finalize_call_id=None,
        authority_kind="v1-capture",
        authority_id="a0",
    )
    base.update(over)
    return module.FactoryEvent(**base)


EXPECTED = {
    "delivery_finalizer/v1": "v1-capture",
    "delivery_finalizer/v2": "optimizer-selection",
    "delivery_finalizer/v2-speaker-off": "optimizer-selection",
}


def ledger_with(module, *events):
    ledger = module.AuthorityLedger()
    for item in events:
        ledger.record(item)
    return ledger


def test_check_roots_passes_for_exactly_the_expected_rows():
    module = authority()
    ledger = ledger_with(
        module,
        event(module),
        event(
            module,
            row_id="delivery_finalizer/v2",
            call_id="c1",
            authority_kind="optimizer-selection",
        ),
        event(
            module,
            row_id="delivery_finalizer/v2-speaker-off",
            call_id="c2",
            authority_kind="optimizer-selection",
        ),
    )
    assert module.check_roots(ledger, expected=EXPECTED) == ()


def test_check_roots_rejects_a_second_call_for_one_row():
    module = authority()
    ledger = ledger_with(module, event(module), event(module, call_id="c1"))
    assert module.check_roots(ledger, expected={"delivery_finalizer/v1": "v1-capture"})


def test_check_roots_rejects_a_non_phase1_input_kind():
    module = authority()
    ledger = ledger_with(module, event(module, input_kind="delivered"))
    assert module.check_roots(ledger, expected={"delivery_finalizer/v1": "v1-capture"})


def test_check_roots_rejects_a_parented_call():
    """No ``FinalizeResult`` may appear in any chain -- a re-finalize is not a root."""
    module = authority()
    ledger = ledger_with(module, event(module, parent_finalize_call_id="c0"))
    assert module.check_roots(ledger, expected={"delivery_finalizer/v1": "v1-capture"})


def test_check_roots_rejects_the_wrong_authority_kind_for_a_row():
    module = authority()
    ledger = ledger_with(module, event(module, authority_kind="optimizer-selection"))
    assert module.check_roots(ledger, expected={"delivery_finalizer/v1": "v1-capture"})


def test_check_roots_rejects_an_unused_laundering_attempt():
    """An event nobody consumed is still a minted root the matrix did not expect."""
    module = authority()
    ledger = ledger_with(
        module, event(module), event(module, row_id="shadow_row", call_id="c9")
    )
    assert module.check_roots(ledger, expected={"delivery_finalizer/v1": "v1-capture"})


def test_check_roots_rejects_an_authority_that_reached_no_root():
    """A minted authority nobody used is a laundering attempt, delivered or not.

    The gate is "did nothing else even try", so the question is not whether the
    second capture produced a stream -- it is that a second capture exists at
    all in a matrix that expects one.
    """
    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()
    capture = fin.capture_v1_reference([cue()], ledger=ledger)
    fin.phase1_from_v1_capture(
        capture,
        profile=profile(),
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    expected = {"delivery_finalizer/v1": "v1-capture"}
    assert module.check_roots(ledger, expected=expected) == ()

    fin.capture_v1_reference([cue(end=2.0)], ledger=ledger)  # never minted from
    problems = module.check_roots(ledger, expected=expected)
    assert any("reached no finalizer root" in problem for problem in problems)


def test_a_second_row_minted_through_the_real_factories_is_still_laundering():
    """The end-to-end shape of the attempt the hand-built events only sketch.

    Nothing here is forged. The second root goes through ``capture_v1_reference``
    and ``phase1_from_v1_capture`` exactly as the first one does, so every seal is
    genuine, the chain resolves, the input kind is ``phase1`` and no call is
    parented. It is a violation for one reason only: the matrix said which rows
    exist, and this is not one of them.

    That is the whole point of gating on the EVENT LEDGER rather than on the
    delivered streams -- a laundered row is indistinguishable from a legitimate
    one at the object level, and only "who was expected" can tell them apart.
    """
    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()
    expected = {"delivery_finalizer/v1": "v1-capture"}

    for row_id in ("delivery_finalizer/v1", "delivery_finalizer/shadow"):
        capture = fin.capture_v1_reference([cue()], ledger=ledger)
        fin.phase1_from_v1_capture(
            capture,
            profile=profile(),
            ledger=ledger,
            row_id=row_id,
            evaluation_id="e0",
        )

    problems = module.check_roots(ledger, expected=expected)
    assert any(
        "unexpected row 'delivery_finalizer/shadow'" in problem for problem in problems
    )
    assert not any("delivery_finalizer/v1" in problem for problem in problems)
    # ... and the same evaluation without the extra row is clean, so the gate is
    # reacting to the laundering and not to anything else in the ledger.
    honest = module.AuthorityLedger()
    fin.phase1_from_v1_capture(
        fin.capture_v1_reference([cue()], ledger=honest),
        profile=profile(),
        ledger=honest,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    assert module.check_roots(honest, expected=expected) == ()


def test_check_roots_resolves_the_seed_chain_through_the_ledger():
    """A root may not declare an authority kind its own chain does not carry.

    The event's ``authority_kind`` is the producer's word for it; the chain is
    the ledger's. Here they disagree -- the row claims an optimizer selection
    while its seed descends from the v1 capture -- and only resolving the chain
    catches it.
    """
    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()
    capture = fin.capture_v1_reference([cue()], ledger=ledger)
    laundered = ledger.issue(
        issuer="test.laundering", kind="optimizer-selection", payload={"cues": []}
    )
    ledger.record(
        module.FactoryEvent(
            evaluation_id="e0",
            row_id="delivery_finalizer/v2",
            call_id=laundered.seal.authority_id,
            input_seed_id=capture.capability.seal.authority_id,
            input_kind="phase1",
            parent_finalize_call_id=None,
            authority_kind="optimizer-selection",
            authority_id=laundered.seal.authority_id,
        )
    )
    problems = module.check_roots(
        ledger, expected={"delivery_finalizer/v2": "optimizer-selection"}
    )
    assert any("seed chain terminating in 'v1-capture'" in p for p in problems)
    assert module.seal_chain(ledger, ledger.events[0])[-1].kind == "v1-capture"


def test_lineage_tuples_are_the_probe_record():
    """The six N8b fields, positionally, sorted -- the shape the artifact carries."""
    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()
    capture = fin.capture_v1_reference([cue()], ledger=ledger)
    stream = fin.phase1_from_v1_capture(
        capture,
        profile=profile(),
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    records = module.lineage_tuples(ledger)
    assert len(module.LINEAGE_FIELDS) == 6
    assert records == (
        (
            "e0",
            "delivery_finalizer/v1",
            stream.seed_id,
            capture.capability.seal.authority_id,
            "phase1",
            None,
        ),
    )


# ------------------------------------------------------------ the factories


def test_v1_capture_seals_at_capture_time():
    """The capture is a deep copy, so a later edit to the caller's list is not it.

    The RED skeleton passed ``profile=None`` here; the factory computes phase 1
    eagerly (a stream of ``Phase1Cue`` is what it returns), so a real profile is
    required -- and having one makes the pin STRONGER, because the minted seed
    can then be read back and shown to carry the sealed 1.0 rather than the 99.0
    written into the caller's dict afterwards.
    """
    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()
    cues = [cue()]
    capture = fin.capture_v1_reference(cues, ledger=ledger)
    cues[0]["end"] = 99.0  # the caller's list is not the sealed payload
    stream = fin.phase1_from_v1_capture(
        capture,
        profile=profile(),
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    assert stream.authority_kind == "v1-capture"
    assert stream.input_kind == "phase1"
    assert stream.cues[0].seed_end == 1.0


def test_finalize_refuses_an_unissued_stream():
    """A hand-rolled stream is structurally perfect and still not a root.

    Nothing about the object is wrong -- the cues are real phase-1 cues taken
    from a legitimate factory run, the seal has the right kind and even the
    right digest. It is rejected because no ledger minted that seal, which is
    the whole difference between an authority and a shape.
    """
    module = authority()
    fin = finalizer()
    assert issubclass(module.UnissuedAuthority, RuntimeError)

    prof = profile()
    ledger = module.AuthorityLedger()
    capture = fin.capture_v1_reference([cue()], ledger=ledger)
    honest = fin.phase1_from_v1_capture(
        capture,
        profile=prof,
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )

    forged_seal = module.Seal(
        issuer="voxweave.core.finalizer.phase1_from_v1_capture",
        authority_id="a99",
        kind="v1-capture",
        digest=honest.capability.seal.digest,
    )
    hand_built = fin.Phase1CueStream(
        cues=honest.cues,
        profile=prof,
        seed_id="a99",
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
        authority_kind="v1-capture",
        capability=module.Capability(seal=forged_seal, ledger=ledger),
    )
    with pytest.raises(module.UnissuedAuthority):
        fin.finalize(
            hand_built,
            profile=prof,
            evidence=fin.FinalizeEvidence(),
            policy=fin.FinalizePolicy(),
        )

    # ... and the honest stream is single-use: one seed, one root.
    fin.finalize(
        honest,
        profile=prof,
        evidence=fin.FinalizeEvidence(),
        policy=fin.FinalizePolicy(),
    )
    with pytest.raises(module.CapabilityConsumed):
        fin.finalize(
            honest,
            profile=prof,
            evidence=fin.FinalizeEvidence(),
            policy=fin.FinalizePolicy(),
        )


def _document():
    """One small en document the optimizer can solve without a v1 reference."""
    from voxweave.core.segdoc import SegDocument, SourceUnit

    prof = profile(max_line_length=12, max_lines=1, min_cue_s=0.5, max_cue_s=7.0)
    spec = [("alpha", 0.0, 0.4), ("beta", 0.5, 0.9), ("gamma", 1.0, 1.4)]
    units = [
        SourceUnit(id=f"u{i}", surface=s, start=a, end=b)
        for i, (s, a, b) in enumerate(spec)
    ]
    return SegDocument(
        language=prof.language,
        units=units,
        profile=prof,
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=" ".join(unit.surface for unit in units),
    )


def test_optimizer_factory_rematerializes_from_the_registered_partition():
    """Never ``result.cues``: a mutable cue list is not an authority."""
    from voxweave.core.boundary_v2 import materialize_cues, optimize_document

    module = authority()
    fin = finalizer()
    document = _document()
    solution = optimize_document(document)
    assert all(interval.adopted is None for interval in solution.solutions)

    ledger = module.AuthorityLedger()
    registered = fin.register_optimizer_selection(solution, ledger=ledger)
    expected = materialize_cues(
        registered.edges,
        registered.atoms,
        document.profile.language,
        fallback_start=registered.fallback_start,
    )
    assert [c["end"] for c in expected] == [
        c["end"] for interval in solution.solutions for c in interval.cues
    ]

    # Launder the selection's own cue dicts in place; the factory must not care.
    for interval in solution.solutions:
        for laundered in interval.cues:
            laundered["end"] = 99.0
            laundered["text"] = "laundered"

    stream = fin.phase1_from_optimizer_selection(
        registered,
        ledger=ledger,
        row_id="delivery_finalizer/v2",
        evaluation_id="e0",
    )
    assert stream.authority_kind == "optimizer-selection"
    assert [c.seed_end for c in stream.cues] == [c["end"] for c in expected]
    assert all("laundered" not in c.text for c in stream.cues)
    assert (
        module.check_roots(
            ledger, expected={"delivery_finalizer/v2": "optimizer-selection"}
        )
        == ()
    )


def test_the_expected_matrix_passes_end_to_end():
    """Both authority kinds in one evaluation: one root each, every seal used.

    The rows share a ledger, which is the shape the harness runs: the gate has to
    tell two legitimate roots from one legitimate root plus a laundered one, and
    it can only do that if every issuance is accounted for.
    """
    from voxweave.core.boundary_v2 import optimize_document

    module = authority()
    fin = finalizer()
    ledger = module.AuthorityLedger()

    capture = fin.capture_v1_reference([cue()], ledger=ledger)
    fin.phase1_from_v1_capture(
        capture,
        profile=profile(),
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    document = _document()
    registered = fin.register_optimizer_selection(
        optimize_document(document), ledger=ledger
    )
    fin.phase1_from_optimizer_selection(
        registered,
        ledger=ledger,
        row_id="delivery_finalizer/v2",
        evaluation_id="e0",
    )

    expected = {
        "delivery_finalizer/v1": "v1-capture",
        "delivery_finalizer/v2": "optimizer-selection",
    }
    assert module.check_roots(ledger, expected=expected) == ()
    assert len(module.lineage_tuples(ledger)) == 2
    assert {record[4] for record in module.lineage_tuples(ledger)} == {"phase1"}
    assert {record[5] for record in module.lineage_tuples(ledger)} == {None}
