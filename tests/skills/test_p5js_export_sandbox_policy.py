"""Regression for #145361: the p5js headless exporter must not disable browser protections.

The exporter used to pass ``--no-sandbox --disable-setuid-sandbox --disable-web-security
--allow-file-access-from-files`` unconditionally. The fix keeps web security on, serves sketch
assets through request interception, and applies the sandbox bypass only for Puppeteer-managed
Chrome under a documented host restriction, with a visible reason.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills/creative/p5js/scripts/export-frames.js"
NODE = shutil.which("node")

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {EventEmitter} = require('node:events');
const source = fs.readFileSync(process.argv[2], 'utf8');

function subject({uid = 1000, docker = false, apparmor = '0', exe = '', configExe = '', native = [],
                  defaultArgsReturnsPromise = false} = {}) {
  const logs = [];
  const io = {
    existsSync: p => docker && p === '/.dockerenv',
    realpathSync: p => p,
    statSync: () => ({isFile: () => true}),
    readFileSync: p => p === '/proc/1/cgroup' ? ''
      : p === '/proc/sys/kernel/apparmor_restrict_unprivileged_userns' ? apparmor
      : Buffer.from('{}'),
  };
  const pptr = {
    default: {configuration: {executablePath: configExe}},
    defaultArgs: o => {
      const args = [...native, ...o.args];
      return defaultArgsReturnsPromise ? Promise.resolve(args) : args;
    },
  };
  const ctx = vm.createContext({
    require: n => n === 'fs' ? io : n === 'puppeteer' ? pptr : require(n),
    process: {env: {PUPPETEER_EXECUTABLE_PATH: exe}, geteuid: () => uid},
    console: {error: (...v) => logs.push(v.join(' '))},
    URL, Buffer, setTimeout, clearTimeout,
  });
  const stop = source.lastIndexOf('\nmain().catch(');
  assert(stop > 0);
  vm.runInContext(source.slice(0, stop) + '\nthis.subject={browserLaunchOptions,serveSketchDir};', ctx);
  return {...ctx.subject, logs};
}

(async () => {
  const cases = [
    [{}, false], [{uid: 0}, true], [{docker: true}, true], [{apparmor: '1'}, true],
    [{uid: 0, exe: '/fixture/chrome'}, false], [{docker: true, configExe: '/fixture/chrome'}, false],
    [{exe: '/fixture/chrome', native: ['--no-sandbox', '--disable-dev-shm-usage']}, true],
    [{docker: true, defaultArgsReturnsPromise: true}, true],
  ];
  for (const [facts, bypass] of cases) {
    const s = subject(facts);
    const o = await s.browserLaunchOptions();
    const args = [...(facts.native || []), ...o.args];
    assert.equal(args.includes('--no-sandbox'), bypass, `no-sandbox for ${JSON.stringify(facts)}`);
    assert.equal(args.filter(a => a === '--no-sandbox').length, bypass ? 1 : 0);
    assert(args.filter(a => a === '--disable-dev-shm-usage').length <= 1);
    assert(!args.includes('--disable-web-security'), 'web security must stay enabled');
    assert(!args.includes('--disable-setuid-sandbox'));
    assert(!args.includes('--allow-file-access-from-files'));
    if (bypass) assert(s.logs.length > 0, 'an automatic bypass must state its reason');
  }
  for (const mode of ['local', 'remote', 'respond-error', 'continue-error']) {
    const s = subject();
    const p = new EventEmitter();
    let response, continued = false, closed = false;
    p.setRequestInterception = async b => assert(b);
    p.isClosed = () => closed;
    p.close = async () => { closed = true; };
    const check = await s.serveSketchDir(p, '/fixture');
    p.emit('request', {
      url: () => (mode === 'local' || mode === 'respond-error')
        ? 'http://p5-sketch.localhost/data.json' : 'https://example.invalid/a',
      respond: async r => { if (mode === 'respond-error') throw new Error('fixture reject'); response = r; },
      continue: async () => { if (mode === 'continue-error') throw new Error('fixture reject'); continued = true; },
    });
    await new Promise(r => setImmediate(r));
    await new Promise(r => setImmediate(r));
    if (mode.endsWith('error')) {
      assert.throws(check, /fixture reject/);
      assert(closed);
    } else {
      check();
      if (mode === 'local') {
        assert.equal(response.status, 200);
        assert.equal(response.contentType, 'application/json');
      } else {
        assert(continued);
      }
    }
  }
  console.log(JSON.stringify({passed: cases.length + 4}));
})().catch(e => { console.error(e); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="node is required to exercise the exporter")
def test_exporter_keeps_browser_security_and_gates_the_sandbox_bypass(tmp_path):
    harness = tmp_path / "verify.cjs"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(harness), str(SCRIPT)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert json.loads(result.stdout.strip().splitlines()[-1])["passed"] == 12
