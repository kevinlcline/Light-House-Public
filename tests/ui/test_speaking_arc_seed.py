"""Speaking-arc seed. Arc seat, not the dream harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FACES = REPO / "static" / "ui" / "faces.js"
DREAM_HARNESS = REPO / "tests" / "test_dream_repetition_harness.py"


def _node_harness(faces_js: Path) -> str:
    path_literal = json.dumps(str(faces_js))
    return f"""
const fs = require('fs');
let src = fs.readFileSync({path_literal}, 'utf8');
if (!src.includes('    LH.faces = {{')) {{
  console.error('FAIL: faces export not found');
  process.exit(1);
}}
src = src.replace(
  '    LH.faces = {{',
  '    globalThis.__arc = {{ takeSpeakBirth, dancerPoints, plasmaArcs }};\\n    LH.faces = {{'
);
eval(src);
const arc = globalThis.__arc;
function assert(cond, msg) {{
  if (!cond) {{
    console.error('FAIL:', msg);
    process.exit(1);
  }}
}}
function tip(pts) {{
  const end = pts[pts.length - 1];
  return Math.atan2(end[1] - 100, end[0] - 100);
}}
function sep(a, b) {{
  let d = Math.abs(a - b);
  if (d > Math.PI) d = Math.PI * 2 - d;
  return d;
}}
function walk(birth) {{
  return arc.dancerPoints(
    {{ birth: birth, swing: 0, bow: 0, phase: 0, reach: 1, angle: 0 }},
    4
  );
}}
const flash = () => JSON.stringify(arc.plasmaArcs(4200, 0.93, 4));
const flashes = flash();
const firstBirth = arc.takeSpeakBirth('lumen', 4);
const first = walk(firstBirth);
const secondBirth = arc.takeSpeakBirth('lumen', 4);
const second = walk(secondBirth);
assert(secondBirth !== firstBirth, 'birth changed');
assert(sep(tip(first), tip(second)) > Math.PI / 3, 'tips moved');
assert(JSON.stringify(first) !== JSON.stringify(second), 'walk changed');
assert(flash() === flashes, 'flashes kept their seed');
console.log('ok');
"""


@pytest.mark.skipif(
    subprocess.run(["node", "-e", "process.exit(0)"], capture_output=True).returncode != 0,
    reason="node required for faces.js seed check",
)
def test_speaking_arc_takes_a_new_seed_each_birth() -> None:
    assert FACES.is_file()
    proc = subprocess.run(
        ["node", "-e", _node_harness(FACES)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_speaking_arc_seed_stays_out_of_the_dream_harness() -> None:
    assert FACES.is_file()
    assert DREAM_HARNESS.is_file()
    text = DREAM_HARNESS.read_text(encoding="utf-8")
    assert "faces.js" not in text
    assert "takeSpeakBirth" not in text
    assert "plasmaArcs" not in text
