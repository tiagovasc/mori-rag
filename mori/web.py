"""Loopback-only research room. This is deliberately not a public SaaS server."""
import base64
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit

from .config import Config
from .ingest import ingest, load_file
from .pipeline import plan, query
from .providers import Embeddings, LLM

UI_FIELDS = {'retrieval_k', 'jev_k', 'final_passages', 'full_units', 'summaries', 'min_relevance',
             'summary_min_relevance', 'summary_candidates', 'rerank_enabled', 'rerank_k',
             'rating_neighbors', 'answer_neighbors', 'exclude_full_unit_passages'}


def serve(store, config, port=8840):
    token = secrets.token_hex(32)
    origin = f'http://127.0.0.1:{port}'
    static = Path(__file__).parent / 'static'
    state, lock = {'status': 'idle'}, threading.Lock()
    embeddings = Embeddings(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, value, kind='application/json'):
            body = json.dumps(value, ensure_ascii=False).encode() if kind == 'application/json' else value
            self.send_response(code)
            for key, item in {'Content-Type': kind + '; charset=utf-8', 'Content-Length': str(len(body)),
                              'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                              'Referrer-Policy': 'no-referrer', 'Cross-Origin-Resource-Policy': 'same-origin',
                              'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"}.items():
                self.send_header(key, item)
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            return self.headers.get('Host') == f'127.0.0.1:{port}'

        def do_GET(self):
            if not self.host_ok():
                return self.send(403, {'error': 'Untrusted Host'})
            path = urlsplit(self.path).path
            if path == '/api/state':
                return self.send(200, {'stats': store.stats(), 'job': state.copy(),
                                      'settings': {k: getattr(config, k) for k in sorted(UI_FIELDS)},
                                      'providers': {'embedding': config.embedding, 'llm': config.llm, 'model': config.llm_model}})
            if path == '/api/runs':
                runs = []
                for item in sorted((store.root / 'runs').glob('*/request.json'), reverse=True)[:100]:
                    request = json.loads(item.read_text(encoding='utf-8'))
                    status_path = item.parent / 'status.json'
                    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'status': 'interrupted'}
                    runs.append({'id': item.parent.name, 'question': request['question'], 'created': request['created'], **status})
                return self.send(200, runs)
            match = re.fullmatch(r'/api/runs/([0-9TZ-]+[a-f0-9]{10})', path)
            if match:
                run = store.root / 'runs' / match[1]
                if not (run / 'context.json').exists():
                    return self.send(404, {'error': 'No completed context for that run'})
                packet = json.loads((run / 'context.json').read_text(encoding='utf-8'))
                return self.send(200, {'context': packet, 'answer': (run / 'answer.md').read_text(encoding='utf-8') if (run / 'answer.md').exists() else '',
                                       'prompt': (run / 'answer_prompt.txt').read_text(encoding='utf-8')})
            if path in ('/', '/app.js', '/styles.css'):
                name = 'index.html' if path == '/' else path[1:]
                content = (static / name).read_bytes()
                if path == '/':
                    content = content.replace(b'__MORI_TOKEN__', token.encode())
                return self.send(200, content, {'index.html': 'text/html', 'app.js': 'application/javascript', 'styles.css': 'text/css'}[name])
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.host_ok() or self.headers.get('Origin') != origin or not secrets.compare_digest(self.headers.get('X-Mori-Token', ''), token):
                return self.send(403, {'error': 'Local request token and exact origin required'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(415, {'error': 'JSON required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 12 * 1024 * 1024:
                    return self.send(413, {'error': 'Upload/request too large'})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Request must be an object')
                path = urlsplit(self.path).path
                if path not in ('/api/query', '/api/upload', '/api/plan'):
                    return self.send(404, {'error': 'Not found'})
                overrides = body.get('settings', {})
                if not isinstance(overrides, dict) or set(overrides) - UI_FIELDS:
                    raise ValueError('Unsupported UI setting')
                active = replace(config, **overrides).validate()
                if path == '/api/plan':
                    return self.send(200, plan(store, active))
                if not lock.acquire(blocking=False):
                    return self.send(409, {'error': 'One operation is already running'})
                if path == '/api/query':
                    question = body.get('question')
                    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
                        lock.release()
                        raise ValueError('Enter a question between 1 and 4000 characters')
                    if type(body.get('answer', False)) is not bool:
                        lock.release()
                        raise ValueError('answer must be a boolean')
                    action = lambda: query(store, question, active, embeddings=embeddings, answer=body.get('answer', False))
                else:
                    try:
                        name = body.get('filename', '')
                        if not isinstance(name, str) or len(name) > 250:
                            raise ValueError('Invalid filename')
                        suffix = Path(name).suffix.lower()
                        if suffix not in ('.jsonl', '.txt', '.md', '.html', '.htm', '.vtt', '.srt', '.docx', '.pdf'):
                            raise ValueError('Unsupported upload type')
                        if body.get('split', 'whole') not in ('whole', 'headings'):
                            raise ValueError('Invalid unit split policy')
                        data = base64.b64decode(body['data'], validate=True)
                        if len(data) > 8 * 1024 * 1024:
                            raise ValueError('Upload exceeds 8 MB')
                        target = store.root / 'uploads' / (secrets.token_hex(12) + suffix)
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(data)
                        def import_upload():
                            units = load_file(target, body.get('split', 'whole'))
                            if suffix != '.jsonl':
                                for index, unit in enumerate(units, 1):
                                    if body.get('split', 'whole') == 'whole':
                                        unit['title'] = Path(name).stem
                                    unit['locator'] = f'{Path(name).name} / unit {index}'
                                    unit['metadata']['filename'] = Path(name).name
                            return ingest(store, units, active, embeddings, LLM(active), bool(body.get('summarize', False)))
                        action = import_upload
                    except Exception:
                        lock.release()
                        raise
                state.clear()
                state.update(status='running')

                def worker():
                    try:
                        result = action()
                        state.update(status='complete', result=result)
                    except (ValueError, RuntimeError) as exc:
                        state.update(status='failed', error=str(exc))
                    except Exception:
                        state.update(status='failed', error='Operation failed; inspect local configuration and source contract')
                    finally:
                        lock.release()
                threading.Thread(target=worker, daemon=True).start()
                return self.send(202, {'status': 'running'})
            except (ValueError, KeyError, TypeError):
                return self.send(400, {'error': 'Invalid request or source/settings contract'})

    print(f'Mori research room: {origin}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
