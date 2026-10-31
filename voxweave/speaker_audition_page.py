"""The self-contained in-memory audition page for diarized speakers."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from voxweave.speaker_snippets import Span
from voxweave.voicebase import html_attribute, html_text
from voxweave.voicematch import MatchCandidate, SpeakerMatch


def _suggestion_button(candidate: MatchCandidate, *, show_scopes: bool) -> str:
    where = ""
    if show_scopes and candidate.scopes:
        where = " from " + ", ".join(candidate.scopes)
    return (
        '<button type="button" class="use-suggestion" '
        f'data-use="{html_attribute(candidate.display_name)}">'
        f"{html_text(candidate.display_name)} ({candidate.similarity:.2f})"
        f"{html_text(where)} [use]"
        "</button>"
    )


def _tiered_suggestions(match: SpeakerMatch | None, scope: str) -> str:
    """Tier 1 (this scope) and tier 2 (other scopes, labelled) suggestion rows."""
    primary = match.candidates if match is not None else ()
    secondary = (
        match.secondary.candidates
        if match is not None and match.secondary is not None
        else ()
    )
    rows = [
        '<div class="suggestions" data-tier="1">'
        f'<span class="tier">This scope ({html_text(scope)})</span>'
        + (
            "".join(_suggestion_button(c, show_scopes=False) for c in primary)
            or '<span class="no-suggestion">No stored match.</span>'
        )
        + "</div>"
    ]
    if secondary:
        rows.append(
            '<div class="suggestions" data-tier="2">'
            '<span class="tier">Other scopes (stricter match)</span>'
            + "".join(_suggestion_button(c, show_scopes=True) for c in secondary)
            + "</div>"
        )
    return "".join(rows)


# The page pieces that depend only on whether suggestions are shown, kept at
# module level so the page function holds just the template.
_MATCHING_CSS = (
    "\n.suggestions { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }"
    "\n.machine-mark, .no-suggestion, .tier { color: #777; font-size: .8rem; }"
    "\n.tier { align-self: center; }"
)
_MATCHING_SCRIPT = (
    "for (const button of document.querySelectorAll('.use-suggestion')) {"
    "\n  button.addEventListener('click', () => {"
    "\n    const field = button.closest('.speaker').querySelector('[data-speaker]');"
    "\n    field.value = button.dataset.use; field.dispatchEvent(new Event('input'));"
    "\n  });"
    "\n}\n"
)
_SPLIT_SCRIPT = """const splitButtons = [...document.querySelectorAll('.split-speaker')];
let splitReady = false;
let splitApplied = false;

function splitErrorMessage(payload, fallback) {
  if (typeof payload?.error === 'string' && payload.error) return payload.error;
  if (typeof payload?.error?.message === 'string' && payload.error.message) return payload.error.message;
  if (typeof payload?.message === 'string' && payload.message) return payload.message;
  return fallback;
}

async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-VoxWeave-Token': sessionToken},
    body: JSON.stringify(payload),
  });
  let responsePayload = {};
  try { responsePayload = await response.json(); } catch (_) {}
  if (!response.ok) {
    const error = new Error(splitErrorMessage(responsePayload, `${path} ${response.status}`));
    error.status = response.status;
    throw error;
  }
  return responsePayload;
}

function validateSplitProposal(proposal, speakerId) {
  if (!proposal || proposal.speaker_id !== speakerId || !proposal.assignment ||
      typeof proposal.assignment !== 'object' || Array.isArray(proposal.assignment) ||
      !Array.isArray(proposal.groups) || proposal.groups.length !== 2) {
    throw new Error('The server returned an invalid split proposal.');
  }
  for (const label of Object.values(proposal.assignment)) {
    if (label !== 'A' && label !== 'B') throw new Error('The server returned an invalid split assignment.');
  }
  for (const group of proposal.groups) {
    if ((group.label !== 'A' && group.label !== 'B') ||
        !Number.isInteger(group.turn_count) || group.turn_count < 1 ||
        typeof group.total_duration !== 'number' || !Number.isFinite(group.total_duration) ||
        !Array.isArray(group.samples)) {
      throw new Error('The server returned an invalid split group.');
    }
    for (const sample of group.samples) {
      if (!sample || typeof sample.start !== 'number' || typeof sample.end !== 'number' ||
          typeof sample.src !== 'string' || !sample.src.startsWith('data:audio/mpeg;base64,')) {
        throw new Error('The server returned an invalid split sample.');
      }
    }
  }
}

function splitStatus(message, className = '') {
  const status = document.createElement('p');
  status.className = className;
  status.textContent = message;
  return status;
}

function setSplitButtonsDisabled(disabled) {
  for (const button of splitButtons) button.disabled = disabled || splitApplied;
}

function finishSplit() {
  splitApplied = true;
  for (const control of document.querySelectorAll(
    '[data-speaker], #save, .use-suggestion, .split-speaker, .split-apply, .split-cancel'
  )) control.disabled = true;
}

// The applied state: the server has ended this session, so the page only
// offers taking the split back (one level, through /split-undo).
function renderSplitApplied(host) {
  const undo = document.createElement('button');
  undo.type = 'button';
  undo.className = 'split-undo';
  undo.textContent = 'Undo this split';
  const actions = document.createElement('div');
  actions.className = 'split-confirm-actions';
  actions.append(undo);
  const error = splitStatus('', 'split-error');
  host.replaceChildren(
    splitStatus('split applied — restart `voxweave speakers` to re-audition'),
    actions,
    error,
  );
  undo.addEventListener('click', async () => {
    undo.disabled = true;
    error.textContent = '';
    try {
      const result = await postJSON('split-undo', {});
      if (!result || result.undone !== true) {
        throw new Error('The server returned an invalid undo confirmation.');
      }
      host.replaceChildren(splitStatus(
        'split undone — the previous speakers are restored; restart `voxweave speakers` to audition again'
      ));
    } catch (requestError) {
      const message = requestError instanceof Error && requestError.message
        ? requestError.message : 'the server did not answer';
      // 409 is a refusal: the server checked before writing, so nothing changed.
      error.textContent = requestError?.status === 409
        ? `${message}. Nothing was restored.`
        : `Undo failed: ${message}`;
      undo.disabled = false;
    }
  });
}

function renderSplitProposal(card, speakerId, proposal) {
  const host = card.querySelector('.split-proposal');
  host.replaceChildren();
  const groups = document.createElement('div');
  groups.className = 'split-groups';
  for (const group of proposal.groups) {
    const section = document.createElement('section');
    section.className = 'split-group';
    const heading = document.createElement('h3');
    heading.textContent = `Group ${group.label}`;
    const summary = document.createElement('p');
    const noun = group.turn_count === 1 ? 'turn' : 'turns';
    summary.textContent = `${group.turn_count} ${noun} · ${group.total_duration.toFixed(1)}s total`;
    section.append(heading, summary);
    for (const sample of group.samples) {
      const figure = document.createElement('figure');
      const caption = document.createElement('figcaption');
      caption.textContent = `${sample.start.toFixed(1)}s- ${sample.end.toFixed(1)}s`;
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'none';
      audio.src = sample.src;
      figure.append(caption, audio);
      section.append(figure);
    }
    groups.append(section);
  }

  const message = splitStatus('Do these groups represent two different people?');
  const error = splitStatus('', 'split-error');
  const actions = document.createElement('div');
  actions.className = 'split-confirm-actions';
  const apply = document.createElement('button');
  apply.type = 'button';
  apply.className = 'split-apply';
  apply.textContent = 'This is correct, apply';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'split-cancel';
  cancel.textContent = 'Cancel';
  actions.append(apply, cancel);
  host.append(message, groups, error, actions);

  cancel.addEventListener('click', () => host.replaceChildren());
  apply.addEventListener('click', async () => {
    apply.disabled = true;
    cancel.disabled = true;
    save.disabled = true;
    setSplitButtonsDisabled(true);
    error.textContent = '';
    try {
      const confirmation = await postJSON('split-confirm', {
        speaker_id: speakerId,
        assignment: proposal.assignment,
      });
      if (!confirmation || typeof confirmation.new_id !== 'string' || !confirmation.new_id) {
        throw new Error('The server returned an invalid split confirmation.');
      }
      finishSplit();
      renderSplitApplied(host);
    } catch (requestError) {
      error.textContent = requestError instanceof Error ? requestError.message : 'Split could not be applied.';
      apply.disabled = false;
      cancel.disabled = false;
      save.disabled = false;
      setSplitButtonsDisabled(false);
    }
  });
}

for (const button of splitButtons) {
  button.addEventListener('click', async () => {
    if (!splitReady || splitApplied) return;
    const card = button.closest('.speaker');
    const host = card.querySelector('.split-proposal');
    const speakerId = button.dataset.split;
    for (const proposal of document.querySelectorAll('.split-proposal')) proposal.replaceChildren();
    host.append(splitStatus('Analyzing this speaker’s turns…'));
    setSplitButtonsDisabled(true);
    try {
      const proposal = await postJSON('split', {speaker_id: speakerId});
      validateSplitProposal(proposal, speakerId);
      renderSplitProposal(card, speakerId, proposal);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Speaker could not be split.';
      host.replaceChildren(splitStatus(message, 'split-error'));
    } finally {
      setSplitButtonsDisabled(false);
    }
  });
}
"""


# One section per speaker; the loop body is moved here unchanged.
def _audition_cards(
    snippets: Mapping[str, Sequence[tuple[Span, str]]],
    matches: Mapping[str, SpeakerMatch] | None,
    scope: str | None,
) -> list[str]:
    cards: list[str] = []
    for speaker_id, clips in snippets.items():
        audio = []
        for (start, end), data_uri in clips:
            audio.append(
                "<figure><figcaption>"
                f"{start:.1f}s- {end:.1f}s"
                f'</figcaption><audio controls preload="none" src="{data_uri}"></audio></figure>'
            )
        if not audio:
            audio.append(
                '<p class="empty">No clean 2-6 second snippet was available.</p>'
            )
        sid = html.escape(speaker_id, quote=True)
        split_controls = (
            '<div class="speaker-actions">'
            '<button type="button" class="split-speaker" '
            f'data-split="{sid}" hidden>Split this speaker</button></div>'
            '<div class="split-proposal" role="status" '
            'aria-live="polite"></div>'
        )
        if matches is None:
            cards.append(
                '<section class="speaker">'
                f'<div class="speaker-head"><code>{html.escape(speaker_id)}</code>'
                f'<input type="text" data-speaker="{sid}" aria-label="Name for {sid}" '
                'placeholder="Enter display name" autocomplete="off"></div>'
                f'{split_controls}<div class="clips">{"".join(audio)}</div></section>'
            )
            continue

        match = matches.get(speaker_id)
        candidates = match.candidates if match is not None else ()
        prefill = (
            candidates[0].display_name
            if match is not None and match.decision == "prefill" and candidates
            else ""
        )
        if scope is not None:
            suggestion_rows = _tiered_suggestions(match, scope)
        else:
            suggestion_buttons = "".join(
                _suggestion_button(candidate, show_scopes=False)
                for candidate in candidates
            )
            if not suggestion_buttons:
                suggestion_buttons = (
                    '<span class="no-suggestion">No stored match.</span>'
                )
            suggestion_rows = f'<div class="suggestions">{suggestion_buttons}</div>'
        machine_mark = (
            '<span class="machine-mark">machine-suggested; review before saving</span>'
            if prefill
            else ""
        )
        cards.append(
            '<section class="speaker">'
            f'<div class="speaker-head"><code>{html.escape(speaker_id)}</code>'
            f'<input type="text" data-speaker="{sid}" aria-label="Name for {sid}" '
            f'value="{html_attribute(prefill)}" placeholder="Enter display name" '
            f'autocomplete="off">{machine_mark}</div>'
            f"{suggestion_rows}"
            f'{split_controls}<div class="clips">{"".join(audio)}</div></section>'
        )
    return cards


def _render_audition_html(
    title: str,
    mapping_name: str,
    snippets: Mapping[str, Sequence[tuple[Span, str]]],
    matches: Mapping[str, SpeakerMatch] | None = None,
    *,
    scope: str | None = None,
) -> str:
    """Build the self-contained in-memory audition page.

    ``scope`` marks a voice-library match: suggestions are then split into the
    episode's own scope and the other scopes, each labelled.
    """
    cards = _audition_cards(snippets, matches, scope)

    safe_title = html.escape(title)
    safe_mapping = html.escape(mapping_name)
    matching_css = _MATCHING_CSS if matches is not None else ""
    matching_script = _MATCHING_SCRIPT if matches is not None else ""
    split_script = _SPLIT_SCRIPT
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title} - speaker audition</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 960px; margin: 0 auto; padding: 2rem; line-height: 1.45; }}
h1 {{ margin-bottom: .25rem; }}
.lede {{ color: #777; margin-top: 0; }}
.speaker {{ border: 1px solid #8886; border-radius: .75rem; padding: 1rem; margin: 1rem 0; }}
.speaker-head {{ display: grid; grid-template-columns: minmax(8rem, 1fr) 2fr; gap: 1rem; align-items: center; }}
input {{ font: inherit; padding: .6rem .75rem; border: 1px solid #8888; border-radius: .4rem; }}
.speaker-actions, .split-confirm-actions {{ display: flex; gap: .5rem; margin-top: .75rem; }}
.split-proposal:empty {{ display: none; }}
.split-proposal {{ border-top: 1px solid #8884; margin-top: 1rem; padding-top: .25rem; }}
.split-groups {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .75rem; }}
.split-group {{ border: 1px solid #8886; border-radius: .5rem; padding: .75rem; }}
.split-group h3, .split-group p {{ margin: 0 0 .5rem; }}
.split-error {{ color: #c33; }}
.clips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .75rem; margin-top: 1rem; }}
figure {{ margin: 0; }} figcaption {{ font-size: .8rem; color: #777; }} audio {{ width: 100%; }}
.empty {{ color: #777; font-style: italic; }}
.save-note:empty {{ display: none; }}
.save-note {{ margin: .5rem 0 0; color: #777; }}
.output {{ position: sticky; bottom: 0; background: Canvas; border: 1px solid #8886; border-radius: .75rem; padding: 1rem; box-shadow: 0 0 2rem #0003; }}
.output-head, .output-actions {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; }}
button {{ font: inherit; padding: .45rem .8rem; cursor: pointer; }}
pre {{ overflow-x: auto; margin-bottom: 0; user-select: all; }}{matching_css}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<p class="lede">Listen, enter names, then save them directly to <code>{safe_mapping}</code>. The Copy button remains available as a fallback.</p>
{"".join(cards)}
<section class="output">
<div class="output-head"><strong>Speaker mapping JSON</strong><span class="output-actions"><button id="save" type="button" hidden>Save</button><button id="copy" type="button">Copy JSON</button></span></div>
<p id="save-note" class="save-note" aria-live="polite"></p>
<pre id="json" aria-live="polite"></pre>
</section>
<script>
const fields = [...document.querySelectorAll('[data-speaker]')];
const output = document.querySelector('#json');
const save = document.querySelector('#save');
const saveNote = document.querySelector('#save-note');
let sessionToken = '';
// Set once the saved names load; empty while Save is unavailable.
let saveLabel = '';
// The mapping as this page last saved it: the button says Saved only while the
// names still match it, so an edit after a save never looks saved.
let savedJSON = null;
function update() {{
  const speakers = {{}};
  for (const field of fields) speakers[field.dataset.speaker] = field.value;
  output.textContent = JSON.stringify({{version: 1, speakers}}, null, 2);
  if (!saveLabel) return;
  const saved = output.textContent === savedJSON;
  save.textContent = saved ? 'Saved' : saveLabel;
  if (!saved) saveNote.textContent = '';
}}
for (const field of fields) field.addEventListener('input', update);
{matching_script}{split_script}document.querySelector('#copy').addEventListener('click', async (event) => {{
  try {{ await navigator.clipboard.writeText(output.textContent); event.target.textContent = 'Copied'; }}
  catch (_) {{ const range = document.createRange(); range.selectNodeContents(output); getSelection().removeAllRanges(); getSelection().addRange(range); event.target.textContent = 'Select and copy'; }}
}});
update();
fetch('serve-info', {{cache: 'no-store'}}).then(async (response) => {{
  if (!response.ok) {{
    let payload = {{}};
    try {{ payload = await response.json(); }} catch (_) {{}}
    throw new Error(splitErrorMessage(payload, `serve-info ${{response.status}}`));
  }}
  const info = await response.json();
  sessionToken = info.token;
  for (const field of fields) {{
    if (Object.prototype.hasOwnProperty.call(info.speakers, field.dataset.speaker)) {{
      field.value = info.speakers[field.dataset.speaker];
    }}
  }}
  saveLabel = `Save to ${{info.mapping_name}}`;
  update();
  save.hidden = false;
  splitReady = true;
  for (const button of splitButtons) button.hidden = false;
}}).catch((error) => {{
  const reason = error instanceof Error && error.message ? `: ${{error.message}}` : '';
  save.textContent = `Could not load the saved names${{reason}} (see the terminal); use Copy JSON`;
  save.disabled = true;
  save.hidden = false;
}});
save.addEventListener('click', async () => {{
  save.disabled = true;
  save.textContent = 'Saving and updating subtitles...';
  const body = output.textContent;
  try {{
    const response = await fetch('save', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json', 'X-VoxWeave-Token': sessionToken}},
      body,
    }});
    if (!response.ok) {{
      let payload = {{}};
      try {{ payload = await response.json(); }} catch (_) {{}}
      throw new Error(splitErrorMessage(payload, ''));
    }}
    let result = {{}};
    try {{ result = await response.json(); }} catch (_) {{}}
    savedJSON = body;
    update();
    if (typeof result.rendered === 'string') {{
      saveNote.textContent = `Subtitles updated: ${{result.rendered}}`;
    }} else if (typeof result.render_error === 'string') {{
      saveNote.textContent = `Subtitles not updated: ${{result.render_error}}`;
    }}
  }} catch (error) {{
    save.textContent = error instanceof Error && error.message ? `Save failed: ${{error.message}}` : 'Save failed';
  }}
  finally {{ save.disabled = splitApplied; }}
}});
</script>
</body>
</html>
"""
