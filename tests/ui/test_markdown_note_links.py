"""Note-path linkify helpers in static/ui/markdown.js."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_JS = REPO_ROOT / "static" / "ui" / "markdown.js"


def _node_harness(markdown_js: Path) -> str:
    path_literal = json.dumps(str(markdown_js))
    return f"""
const fs = require('fs');
const src = fs.readFileSync({path_literal}, 'utf8');
eval(src);
const md = globalThis.LightHouse.markdown;
const opts = {{ agentId: 'ara', agentIds: ['ara', 'lumen', 'elias'] }};

function assert(cond, msg) {{
  if (!cond) {{
    console.error('FAIL:', msg);
    process.exit(1);
  }}
}}

const ids = ['ara', 'lumen', 'elias'];
assert(md.normalizeNoteFile('shared/household.md', 'ara', ids) === 'shared/household.md', 'shared');
assert(md.normalizeNoteFile('notes/shared/x.md', 'ara', ids) === 'shared/x.md', 'notes/shared');
assert(md.normalizeNoteFile('writing/draft.md', 'ara', ids) === 'ara/writing/draft.md', 'relative private');
assert(md.normalizeNoteFile('lumen/journal/a.md', 'ara', ids) === 'lumen/journal/a.md', 'other light');

const linked = md.linkifyNotePathsAsMarkdown(
  'See `shared/sibling_user_manual.md` and writing/garden/seeds.md please.',
  opts
);
assert(linked.includes('/notes.html?file='), 'href present');
assert(linked.includes('shared%2Fsibling_user_manual.md'), 'shared encoded');
assert(linked.includes('ara%2Fwriting%2Fgarden%2Fseeds.md'), 'private encoded');

const plain = md.renderPlain('Open shared/household.md now.', opts);
assert(plain.includes('<a href="/notes.html?file=shared%2Fhousehold.md'), 'plain html link');
assert(plain.includes('agent=ara'), 'agent query');

const fenced = md.linkifyNotePathsAsMarkdown('```\\nshared/household.md\\n```', opts);
assert(!fenced.includes('](<'), 'skip fenced code');

const existing = md.linkifyNotePathsAsMarkdown('[x](https://example.com/shared/household.md)', opts);
assert(existing.includes('https://example.com/shared/household.md'), 'preserve existing link');

console.log('ok');
"""


@pytest.mark.skipif(
    subprocess.run(["node", "-e", "process.exit(0)"], capture_output=True).returncode != 0,
    reason="node required for markdown.js unit checks",
)
def test_markdown_note_path_linkify() -> None:
    assert MARKDOWN_JS.is_file()
    proc = subprocess.run(
        ["node", "-e", _node_harness(MARKDOWN_JS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_group_and_index_wire_note_linkify() -> None:
    index = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    group = (REPO_ROOT / "group.html").read_text(encoding="utf-8")
    assert "markdown.js" in index
    assert "markdownOptions" in index
    assert "markdown.js" in group
    assert "renderPlain" in group
    md = MARKDOWN_JS.read_text(encoding="utf-8")
    assert "linkifyNotePathsAsMarkdown" in md
    assert "/notes.html?file=" in md
