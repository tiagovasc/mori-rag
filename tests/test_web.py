"""Real loopback server checks; no provider calls or external network."""
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / '.local' / 'web-tests'
        directory.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=directory)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.origin = f'http://127.0.0.1:{cls.port}'
        env = {k: v for k, v in os.environ.items() if k != 'TYPESAFE_API_KEY'}
        cls.process = subprocess.Popen([sys.executable, '-m', 'mori', '--config', 'examples/demo.toml',
                                      '--data', cls.temp.name, 'serve', '--port', str(cls.port)],
                                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.origin, timeout=.3) as response:
                    cls.html = response.read().decode()
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.05)
        else:
            cls.process.terminate()
            raise RuntimeError('Test server did not become reachable')
        cls.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', cls.html)[1]

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.temp.cleanup()

    def request(self, path, body=None, headers=None):
        h = {'Origin': self.origin, 'X-Mori-Token': self.token, 'Content-Type': 'application/json'}
        h.update(headers or {})
        req = urllib.request.Request(self.origin + path, data=None if body is None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
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
