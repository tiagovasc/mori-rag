"""Real loopback server checks; no provider calls or external network."""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class UIBootTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the UI startup regression')
    def test_controls_wait_for_initial_job_state(self):
        # Exercise the actual script with a deliberately delayed server response.
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element() {
  return {disabled: false, append() {}, replaceChildren() {}, setAttribute() {}, addEventListener() {}};
}
const document = {
  getElementById(id) {if (!elements.has(id)) elements.set(id, element()); return elements.get(id);},
  createElement: element,
  querySelector() {return {content: 'synthetic-token'};},
};
let releaseState;
const state = new Promise(resolve => {releaseState = resolve;});
let stateRequests = 0;
const fetch = async path => {
  if (path === '/api/state') {stateRequests++; return {ok: true, json: () => state};}
  assert.equal(path, '/api/runs');
  return {ok: true, json: async () => []};
};
const context = vm.createContext({document, fetch, setTimeout});
vm.runInContext(fs.readFileSync('mori/static/app.js', 'utf8'), context);
for (const id of ['upload', 'retrieve', 'answer', 'plan']) assert.equal(elements.get(id).disabled, true, id);
releaseState({stats: {units: 1, chunks: 1}, sources: [{id: 'fixture', label: 'Fixture', units: 1, chunks: 1}], allow_uploads: true, settings: {}, providers: {embedding: 'demo', llm: 'none'}, job: {status: 'idle'}});
setImmediate(() => {
  for (const id of ['upload', 'retrieve', 'answer', 'plan']) assert.equal(elements.get(id).disabled, false, id);
  assert.equal(stateRequests, 1);
});
"""
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / '.local' / 'web-tests'
        directory.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=directory)
        cls.addClassCleanup(cls.temp.cleanup)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.origin = f'http://127.0.0.1:{cls.port}'
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        env = {k: v for k, v in os.environ.items() if k != 'TYPESAFE_API_KEY'}
        cls.process = subprocess.Popen([sys.executable, '-m', 'mori', '--config', 'examples/demo.toml',
                                      '--data', cls.temp.name, 'serve', '--port', str(cls.port)],
                                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        def stop_server():
            if cls.process.poll() is None:
                cls.process.terminate()
            cls.process.communicate(timeout=10)
        cls.addClassCleanup(stop_server)
        last_error = None
        for _ in range(100):
            if cls.process.poll() is not None:
                _, errors = cls.process.communicate(timeout=10)
                raise RuntimeError('Test server exited before startup: ' + errors.decode(errors='replace'))
            try:
                with cls.opener.open(cls.origin, timeout=.3) as response:
                    cls.html = response.read().decode()
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(.05)
        else:
            raise RuntimeError(f'Test server did not become reachable: {last_error}')
        cls.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', cls.html)[1]

    def request(self, path, body=None, headers=None):
        h = {'Origin': self.origin, 'X-Mori-Token': self.token, 'Content-Type': 'application/json'}
        h.update(headers or {})
        req = urllib.request.Request(self.origin + path, data=None if body is None else json.dumps(body).encode(), headers=h)
        try:
            with self.opener.open(req, timeout=3) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as response:
            return response.code, response.read(), response.headers

    def test_host_origin_token_and_content_type(self):
        for header in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}, {'X-Mori-Token': ''}):
            self.assertEqual(self.request('/api/plan', {}, header)[0], 403)
        self.assertEqual(self.request('/api/plan', {}, {'Content-Type': 'text/plain'})[0], 415)

    def test_provider_overrides_and_invalid_settings_denied(self):
        for settings in ({'llm_command': ['malicious']}, {'workers': 50}, {'retrieval_k': -1}, {'rerank_enabled': 'false'}):
            self.assertEqual(self.request('/api/plan', {'settings': settings})[0], 400)

    def test_readonly_plan_does_not_need_credentials(self):
        code, body, _ = self.request('/api/plan', {})
        self.assertEqual(code, 200)
        self.assertIn('maximum_rating_decisions', json.loads(body))

    def test_upload_requires_explicit_boolean_summary_consent(self):
        uploads = Path(self.temp.name) / 'uploads'
        before = set(uploads.glob('*'))
        for value in ('false', 'true', 0, 1, None, []):
            code, _, _ = self.request('/api/upload', {
                'filename': 'untrusted.txt', 'data': base64.b64encode(b'Synthetic source.').decode(),
                'summarize': value,
            })
            self.assertEqual(code, 400, value)
        self.assertEqual(set(uploads.glob('*')), before)
        self.assertEqual(self.request('/api/plan', {})[0], 200)

    def test_static_allowlist_security_headers(self):
        code, _, headers = self.request('/styles.css')
        self.assertEqual(code, 200)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('/../corpus.sqlite')[0], 404)
        self.assertEqual(self.request('/.env')[0], 404)
        self.assertEqual(self.request('/api/runs/../../env')[0], 404)

    def test_upload_embeds_explicit_units_and_preserves_sources(self):
        data = (ROOT / 'examples/corpus.jsonl').read_bytes()
        code, _, _ = self.request('/api/upload', {'filename': '../../fixture.jsonl', 'data': base64.b64encode(data).decode()})
        self.assertEqual(code, 202)
        for _ in range(100):
            _, body, _ = self.request('/api/state')
            state = json.loads(body)
            if state['job']['status'] != 'running':
                break
            time.sleep(.05)
        self.assertEqual(state['job']['status'], 'complete', state['job'])
        self.assertEqual(state['stats']['units'], 7)
        self.assertFalse((Path(self.temp.name).parent / 'fixture.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
