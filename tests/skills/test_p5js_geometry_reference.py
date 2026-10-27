"""Regression for #145269: the p5js shapes-and-geometry sdSegment example must
return the point distance for strictly coincident endpoints instead of
evaluating the undefined 0/0 projection."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "creative"
    / "p5js"
    / "references"
    / "shapes-and-geometry.md"
)
NODE = shutil.which("node")

# Runs the function extracted from the reference doc, never a rewritten copy, with elementary
# doubles standing in for the p5.js vector/dist helpers the example calls.
_HARNESS = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');

class VectorDouble {
  constructor(x, y) { this.x = x; this.y = y; }
  dot(v) { return this.x * v.x + this.y * v.y; }
  static add(a, b) { return new VectorDouble(a.x + b.x, a.y + b.y); }
  static mult(v, s) { return new VectorDouble(v.x * s, v.y * s); }
}
const sandbox = {
  createVector: (x, y) => new VectorDouble(x, y),
  p5: { Vector: VectorDouble },
  constrain: (v, lo, hi) => Math.max(Math.min(v, hi), lo),
  dist: (px, py, qx, qy) => Math.hypot(px - qx, py - qy),
};
vm.createContext(sandbox);
new vm.Script(SOURCE + '\nthis.sdSegment = sdSegment;').runInContext(sandbox, { timeout: 1000 });

// Expected values are elementary point/segment geometry, not calls to the subject.
const cases = [
  ['degenerate-origin', [3, 4, 0, 0, 0, 0], 5],
  ['degenerate-negative-point', [-5, -6, -2, -2, -2, -2], 5],
  ['degenerate-coincident-point', [-2, 7, -2, 7, -2, 7], 0],
  ['degenerate-fractional-distance', [0.3, 0.4, 0, 0, 0, 0], 0.5],
  ['degenerate-large-finite-distance', [1e200, 1e200, 0, 0, 0, 0], Math.SQRT2 * 1e200],
  ['inside-horizontal', [5, 3, 0, 0, 10, 0], 3],
  ['outside-after-b', [13, -4, 0, 0, 10, 0], 5],
];
for (const [name, args, expected] of cases) {
  const actual = sandbox.sdSegment(...args);
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-12 * Math.max(1, expected),
    `${name}: expected=${expected} actual=${actual}`,
  );
}
console.log(`sdSegment cases passed=${cases.length}`);
"""


@pytest.mark.skipif(NODE is None, reason="node is required to execute the p5.js example")
def test_sdsegment_handles_coincident_endpoints():
    matches = re.findall(
        r"^function sdSegment\(px, py, ax, ay, bx, by\) \{\n[\s\S]*?^\}",
        REFERENCE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert len(matches) == 1, f"expected exactly one sdSegment example, found {len(matches)}"

    harness = f"const SOURCE = {json.dumps(matches[0])};\n" + _HARNESS
    result = subprocess.run(
        [NODE, "-e", harness], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
