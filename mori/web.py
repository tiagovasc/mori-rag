"""Loopback-only research room. This is deliberately not a public SaaS server."""
import base64
import binascii
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, urlencode, urlsplit

from .config import Config
from .budget import COUNTING_METHOD, COUNTING_NOTE
from .ingest import ingest, load_file
from .live import LiveBoard
from .pipeline import STOPPED_MESSAGE, Cancelled, plan, query
from .providers import Embeddings, LLM
from .progress import Progress, UPLOAD_STEPS
from .query_expansion import KINDS, MAX_VARIANT_CHARS
from .store import atomic_json, now

UI_FIELDS = {'retrieval_k', 'jev_k', 'final_passages', 'full_units', 'summaries', 'min_relevance',
             'summary_min_relevance', 'summary_candidates', 'rerank_enabled', 'rerank_k',
             'rating_neighbors', 'answer_neighbors', 'exclude_full_unit_passages', 'llm_model',
             'evidence_mode', 'context_window_tokens', 'llm_max_output_tokens', 'query_expansion'}


def saved_search_phrasings(run, question):
    """Expose only bounded active retrieval hints, never provider diagnostics."""
    try:
        path = run / 'query_expansion.json'
        if path.stat().st_size > 256000:
            return []
        saved = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return []
    if not isinstance(saved, dict) or saved.get('status') not in ('complete', 'partial', 'fallback'):
        return []
    variants = saved.get('variants')
    if not isinstance(variants, list) or not 1 <= len(variants) <= 5:
        return []
    result, seen = [], set()
    for item in variants:
        if not isinstance(item, dict):
            continue
        kind, text = item.get('kind'), item.get('text')
        if (not isinstance(kind, str) or kind not in ('original', *KINDS) or kind in seen
                or not isinstance(text, str) or not text.strip() or len(text) > MAX_VARIANT_CHARS
                or (kind == 'original' and text != question)):
            continue
        seen.add(kind)
        result.append({'kind': kind, 'text': text})
    return result if result and result[0]['kind'] == 'original' else []


FONT_FILES = frozenset({'literata-latin.woff2', 'literata-latin-ext.woff2',
                        'source-sans-3-latin.woff2', 'source-sans-3-latin-ext.woff2'})
VERDICTS = ('yes', 'partly', 'no')
CITATION = re.compile(r'\[([UMP][1-9][0-9]{0,3})\]')
SUMMARY_SETTINGS = ('retrieval_k', 'jev_k', 'final_passages', 'full_units', 'summaries', 'rerank_enabled',
                    'rerank_k', 'evidence_mode', 'llm_model')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_summary(request_path, verdicts=None):
    """List one saved inquiry. An unreadable run must never hide the others."""
    try:
        request = _read_json(request_path)
    except (OSError, ValueError):
        return None
    if not isinstance(request, dict) or not isinstance(request.get('question'), str):
        return None
    try:
        status = _read_json(request_path.parent / 'status.json')
        if not isinstance(status, dict):
            status = {'status': 'unreadable'}
    except FileNotFoundError:
        status = {'status': 'interrupted'}
    except (OSError, ValueError):
        status = {'status': 'unreadable'}
    row = {'id': request_path.parent.name, 'question': request['question'],
           'created': request.get('created'), **status}
    sources = request.get('sources')
    if isinstance(sources, list):
        row['sources'] = [str(item.get('label') or item.get('id')) for item in sources
                          if isinstance(item, dict) and (item.get('label') or item.get('id'))][:50]
    settings = request.get('config')
    if isinstance(settings, dict):
        row['settings'] = {key: settings[key] for key in SUMMARY_SETTINGS if key in settings}
    if status.get('status') == 'complete' and status.get('answer') is True:
        try:
            answer = request_path.parent / 'answer.md'
            if answer.stat().st_size <= 2_000_000:
                cited = CITATION.findall(answer.read_text(encoding='utf-8'))
                row['citations'] = len(cited)
                row['cited'] = len(set(cited))
                if isinstance(verdicts, dict):
                    row['checked'] = len(set(cited) & set(verdicts))
        except (OSError, UnicodeError):
            pass
    return row


def run_meta(run):
    """Display facts for a saved inquiry: when it ran and which permitted model it used."""
    meta = {}
    try:
        request = _read_json(run / 'request.json')
    except (OSError, ValueError):
        request = None
    if isinstance(request, dict):
        settings = request.get('config') if isinstance(request.get('config'), dict) else {}
        if isinstance(request.get('created'), str):
            meta['created'] = request['created']
        if isinstance(settings.get('llm_model'), str):
            meta['llm_model'] = settings['llm_model']
        if settings.get('evidence_mode') in ('fixed', 'context'):
            meta['evidence_mode'] = settings['evidence_mode']
    try:
        status = _read_json(run / 'status.json')
    except (OSError, ValueError):
        status = None
    if isinstance(status, dict) and type(status.get('answer')) is bool:
        meta['answer'] = status['answer']
    return meta


def create_server(store, config, port=8840, *, allow_uploads=True):
    """Bind a loopback server without starting its event loop."""
    if type(allow_uploads) is not bool:
        raise ValueError('allow_uploads must be a boolean')
    config.validate()
    model_choices = config.available_llm_models()
    permitted_models = set(model_choices) | ({''} if not config.llm_model else set())
    token = secrets.token_hex(32)
    origin = f'http://127.0.0.1:{port}'
    static = Path(__file__).parent / 'static'
    state, lock = {'status': 'idle'}, threading.Lock()
    state_lock = threading.Lock()
    embedding_provider = None
    # The live board holds source excerpts: memory only, replaced by the next job, never persisted.
    live = {'board': None, 'cancel': None}
    verdict_lock = threading.Lock()
    room_root = getattr(store, 'root', None)
    verdict_root = Path(room_root) / 'verdicts' if isinstance(room_root, (str, os.PathLike)) else None
    features = ['live', 'cancel', 'fonts'] + (['verdicts'] if verdict_root else []) + (
        ['unit_text'] if callable(getattr(store, 'unit_text', None)) else [])

    def read_verdicts(run_id):
        if verdict_root is None:
            return {}
        try:
            saved = _read_json(verdict_root / f'{run_id}.json')
        except (OSError, ValueError):
            return {}
        rows = saved.get('verdicts') if isinstance(saved, dict) else None
        if not isinstance(rows, dict):
            return {}
        return {key: value['verdict'] for key, value in rows.items()
                if isinstance(key, str) and CITATION.fullmatch(f'[{key}]') and isinstance(value, dict)
                and value.get('verdict') in VERDICTS}

    def find_run(run_id):
        return next((root / 'runs' / run_id for root in run_roots() if (root / 'runs' / run_id).is_dir()), None)

    def job_snapshot():
        with state_lock:
            return deepcopy(state)

    def update_progress(snapshot):
        # Publish terminal success together with its result, never as a running
        # job displaying 100%. No model or operation lock is held by readers.
        if snapshot['percent'] != 100:
            with state_lock:
                state['progress'] = snapshot

    def embeddings():
        nonlocal embedding_provider
        if embedding_provider is None:
            embedding_provider = Embeddings(config)
        return embedding_provider

    def run_roots():
        roots = store.run_roots() if hasattr(store, 'run_roots') else [store.root]
        return list(dict.fromkeys(Path(root).resolve() for root in roots))

    def saved_runs():
        runs = {}
        for root in run_roots():
            for item in sorted((root / 'runs').glob('*/request.json')):
                if re.fullmatch(r'[0-9TZ-]+[a-f0-9]{10}', item.parent.name):
                    runs.setdefault(item.parent.name, item)
        return runs

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, value, kind='application/json', headers=None, cache='no-store'):
            body = json.dumps(value, ensure_ascii=False).encode() if kind == 'application/json' else value
            self.send_response(code)
            content_type = kind if kind.startswith('font/') else kind + '; charset=utf-8'
            for key, item in {'Content-Type': content_type, 'Content-Length': str(len(body)),
                              'Cache-Control': cache, 'X-Content-Type-Options': 'nosniff',
                              'Referrer-Policy': 'no-referrer', 'Cross-Origin-Resource-Policy': 'same-origin',
                              'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"}.items():
                self.send_header(key, item)
            for key, item in (headers or {}).items():
                self.send_header(key, item)
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            return self.headers.get('Host') == f'127.0.0.1:{port}'

        def do_GET(self):
            if not self.host_ok():
                return self.send(403, {'error': 'Untrusted Host'})
            parts = urlsplit(self.path)
            path = parts.path
            if path == '/api/live':
                # Fast poll for the progress screen: the job and, for queries, the in-memory rating board.
                with state_lock:
                    job, board = deepcopy(state), live['board']
                return self.send(200, {'job': job, 'live': board.snapshot() if board is not None else None})
            if path == '/api/unit':
                values = parse_qs(parts.query, keep_blank_values=True).get('id', [])
                if 'unit_text' not in features:
                    return self.send(404, {'error': 'This room cannot show source text'})
                if len(values) != 1 or not 1 <= len(values[0]) <= 1000:
                    return self.send(400, {'error': 'Ask for exactly one source unit'})
                unit = store.unit_text(values[0])
                if not isinstance(unit, dict) or not isinstance(unit.get('text'), str):
                    return self.send(404, {'error': 'That source unit is not in this room'})
                if 'collection' not in unit:
                    unit['collection'] = next((row['label'] for row in store.source_catalog()
                                               if row.get('id') == unit.get('source_id')), None)
                keys = ('id', 'source_id', 'title', 'kind', 'locator', 'text', 'summary', 'metadata', 'passages', 'collection')
                return self.send(200, {key: unit[key] for key in keys if key in unit})
            if path.startswith('/fonts/') and path[len('/fonts/'):] in FONT_FILES:
                font = static / 'fonts' / path[len('/fonts/'):]
                if font.is_file():
                    return self.send(200, font.read_bytes(), 'font/woff2', cache='public, max-age=604800')
            if path == '/api/state':
                return self.send(200, {'stats': store.stats(), 'job': job_snapshot(), 'features': features,
                                      'sources': store.source_catalog(), 'allow_uploads': allow_uploads,
                                      'settings': {k: getattr(config, k) for k in sorted(UI_FIELDS)},
                                      'context_budget': {'counting_method': COUNTING_METHOD,
                                                         'counting_note': COUNTING_NOTE,
                                                         'overhead_reserve_tokens': config.context_overhead_tokens},
                                      'providers': {'embedding': config.embedding, 'llm': config.llm, 'model': config.llm_model,
                                                    'models': model_choices}})
            if path == '/api/runs':
                runs = [row for name, item in sorted(saved_runs().items(), reverse=True)[:100]
                        if (row := run_summary(item, read_verdicts(name))) is not None]
                with state_lock:
                    board = live['board'] if state.get('status') == 'running' and state.get('kind') == 'query' else None
                running = board.run_id() if board is not None and callable(getattr(board, 'run_id', None)) else None
                for row in runs:
                    # The folder of the inquiry still running has no status yet; it is not an interrupted one.
                    if running and row['id'] == running and row.get('status') == 'interrupted':
                        row['status'] = 'running'
                return self.send(200, runs)
            match = re.fullmatch(r'/api/runs/([0-9TZ-]+[a-f0-9]{10})', path)
            if match:
                run = find_run(match[1])
                if run is None or not (run / 'context.json').exists():
                    return self.send(404, {'error': 'No completed context for that run'})
                try:
                    packet = _read_json(run / 'context.json')
                    if not isinstance(packet, dict):
                        raise ValueError('Saved context must be an object')
                    answer = (run / 'answer.md').read_text(encoding='utf-8') if (run / 'answer.md').exists() else ''
                    # Exact bytes: the handoff must match what was counted and sent.
                    prompt_path = run / 'answer_prompt.txt'
                    prompt = prompt_path.read_bytes().decode('utf-8') if prompt_path.exists() else ''
                except (OSError, ValueError):
                    return self.send(500, {'error': 'This saved inquiry could not be read; its files may be incomplete'})
                return self.send(200, {'context': packet, 'answer': answer, 'prompt': prompt,
                                       'search_phrasings': saved_search_phrasings(run, packet.get('question')),
                                       'meta': run_meta(run), 'verdicts': read_verdicts(match[1])})
            if path == '/open':
                values = parse_qs(parts.query, keep_blank_values=True).get('collection', [])
                if len(values) != 1 or values[0] not in {row['id'] for row in store.source_catalog()}:
                    return self.send(404, {'error': 'Unknown source selection'})
                return self.send(302, b'', 'text/plain', {'Location': '/?' + urlencode({'sources': values[0]})})
            if path in ('/', '/app.js', '/styles.css'):
                name = 'index.html' if path == '/' else path[1:]
                content = (static / name).read_bytes()
                if path == '/':
                    content = content.replace(b'__MORI_TOKEN__', token.encode())
                return self.send(200, content, {'index.html': 'text/html', 'app.js': 'application/javascript', 'styles.css': 'text/css'}[name])
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.host_ok() or self.headers.get('Origin') != origin or not secrets.compare_digest(self.headers.get('X-Mori-Token', ''), token):
                return self.send(403, {'error': 'This page is out of date or did not come from this research room. '
                                                'Reload the page and try again (local page token and exact origin required).'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(415, {'error': 'JSON required'})
            try:
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    raise ValueError('Invalid Content-Length') from None
                if not 0 < length <= 12 * 1024 * 1024:
                    return self.send(413, {'error': 'Upload/request too large'})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Request must be an object')
                path = urlsplit(self.path).path
                if path == '/api/cancel':
                    with state_lock:
                        event = live['cancel'] if state.get('status') == 'running' and state.get('kind') == 'query' else None
                        if event is not None:
                            event.set()
                            state['stopping'] = True
                    if event is None:
                        return self.send(409, {'error': 'No inquiry is running.'})
                    return self.send(202, {'status': 'stopping'})
                verdict_match = re.fullmatch(r'/api/runs/([0-9TZ-]+[a-f0-9]{10})/verdicts', path)
                if verdict_match:
                    return self.save_verdict(verdict_match[1], body)
                if path not in ('/api/query', '/api/upload', '/api/plan'):
                    return self.send(404, {'error': 'Not found'})
                if path == '/api/upload' and not allow_uploads:
                    return self.send(403, {'error': 'Uploads are disabled for this research room'})
                overrides = body.get('settings', {})
                if not isinstance(overrides, dict) or set(overrides) - UI_FIELDS:
                    raise ValueError('Unsupported UI setting')
                if 'llm_model' in overrides and (not isinstance(overrides['llm_model'], str)
                                                or overrides['llm_model'] not in permitted_models):
                    raise ValueError('Unsupported LLM model')
                if overrides.get('query_expansion') is True and config.llm == 'none':
                    return self.send(400, {'error': 'Prompt extension requires a configured LLM.'})
                active = replace(config, **overrides).validate()
                selected = store.select_sources(body['sources']) if path in ('/api/query', '/api/plan') and 'sources' in body else store
                if path == '/api/plan':
                    return self.send(200, plan(selected, active))
                # Validate the whole request before taking the operation lock or writing files.
                if path == '/api/query':
                    question = body.get('question')
                    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
                        raise ValueError('Enter a question between 1 and 4000 characters')
                    if type(body.get('answer', False)) is not bool:
                        raise ValueError('answer must be a boolean')
                    if config.llm == 'none' and (body.get('answer', False) or active.rerank_enabled):
                        raise ValueError('No answer model is configured for this research room. Build evidence '
                                         'without reranking, then use the Model handoff prompt with your own LLM.')
                else:
                    name = body.get('filename', '')
                    if not isinstance(name, str) or len(name) > 250:
                        raise ValueError('Invalid filename')
                    suffix = Path(name).suffix.lower()
                    if suffix not in ('.jsonl', '.txt', '.md', '.html', '.htm', '.vtt', '.srt', '.docx', '.pdf'):
                        raise ValueError('Unsupported upload type')
                    split = body.get('split', 'whole')
                    if split not in ('whole', 'headings'):
                        raise ValueError('Invalid unit split policy')
                    summarize = body.get('summarize', False)
                    if type(summarize) is not bool:
                        raise ValueError('summarize must be a boolean')
                    if not isinstance(body.get('data'), str):
                        raise ValueError('The uploaded file data is missing')
                    try:
                        data = base64.b64decode(body['data'], validate=True)
                    except (binascii.Error, ValueError):
                        raise ValueError('The uploaded file data is not valid base64') from None
                    if len(data) > 8 * 1024 * 1024:
                        raise ValueError('Upload exceeds 8 MB')
                if not lock.acquire(blocking=False):
                    return self.send(409, {'error': 'One operation is already running'})
                started = False
                try:
                    board, cancel = None, None
                    if path == '/api/query':
                        board, cancel = LiveBoard(question, active.rubric), threading.Event()
                        action = lambda: query(selected, question, active, embeddings=embeddings(),
                                               answer=body.get('answer', False), progress=progress,
                                               live=board, cancel=cancel)
                    else:
                        target = store.root / 'uploads' / (secrets.token_hex(12) + suffix)
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(data)

                        def import_upload():
                            progress.complete('prepare')
                            progress.start('read', message='Extracting the supplied file into source units.')
                            units = load_file(target, split)
                            if suffix != '.jsonl':
                                for index, unit in enumerate(units, 1):
                                    if split == 'whole':
                                        unit['title'] = Path(name).stem
                                    unit['locator'] = f'{Path(name).name} / unit {index}'
                                    unit['metadata']['filename'] = Path(name).name
                            progress.complete('read')
                            progress.start('import', message='Preparing embeddings and importing units. Per-unit progress is not available for this step.')
                            result = ingest(store, units, active, embeddings(), LLM(active), summarize)
                            progress.complete('import')
                            progress.start('save', message='Confirming the stored index is readable.')
                            store.stats()
                            progress.complete('save')
                            progress.finish()
                            return result
                        action = import_upload
                    progress = Progress(update_progress) if path == '/api/query' else Progress(update_progress, UPLOAD_STEPS)
                    initial_progress = progress.snapshot()
                    with state_lock:
                        state.clear()
                        state.update(status='running', kind='query' if path == '/api/query' else 'upload',
                                     progress=initial_progress)
                        live.update(board=board, cancel=cancel)
                        if path == '/api/query':
                            # Display facts only: the person's own question and request, never source text.
                            state['request'] = {'question': question, 'answer': body.get('answer', False),
                                                'sources': body['sources'] if isinstance(body.get('sources'), list) else None,
                                                'settings': {key: getattr(active, key) for key in SUMMARY_SETTINGS}}
                        else:
                            state['request'] = {'filename': Path(name).name}
                    progress.start('prepare', message='Preparing the operation and starting configured providers.')

                    def worker():
                        try:
                            result = action()
                            final_progress = progress.snapshot()
                            with state_lock:
                                state.update(status='complete', result=result, progress=final_progress)
                        except Cancelled:
                            final_progress = progress.snapshot()
                            with state_lock:
                                state.update(status='stopped', error=STOPPED_MESSAGE, progress=final_progress)
                        except (ValueError, RuntimeError) as exc:
                            progress.fail()
                            final_progress = progress.snapshot()
                            with state_lock:
                                state.update(status='failed', error=str(exc), progress=final_progress)
                        except Exception:
                            progress.fail()
                            final_progress = progress.snapshot()
                            with state_lock:
                                state.update(status='failed', error='Operation failed; inspect local configuration and source contract',
                                             progress=final_progress)
                        finally:
                            lock.release()
                    threading.Thread(target=worker, daemon=True).start()
                    started = True
                finally:
                    # Never leave the room permanently busy when the job could not start.
                    if not started:
                        lock.release()
                return self.send(202, {'status': 'running'})
            except json.JSONDecodeError:
                return self.send(400, {'error': 'The request body must be valid JSON'})
            except ValueError as exc:
                # Contract messages are written for people and never contain provider payloads.
                return self.send(400, {'error': str(exc) or 'Invalid request'})
            except (KeyError, TypeError):
                return self.send(400, {'error': 'Invalid request or source/settings contract'})

    def save_verdict(handler, run_id, body):
        """Record the person's check of one citation, outside the saved inquiry so its files never change."""
        if verdict_root is None:
            return handler.send(404, {'error': 'This room cannot store citation checks'})
        citation, verdict = body.get('citation'), body.get('verdict')
        if not isinstance(citation, str) or not CITATION.fullmatch(f'[{citation}]'):
            raise ValueError('Choose a citation such as P1')
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError('A check must be yes, partly, no or cleared')
        run = find_run(run_id)
        if run is None or not (run / 'context.json').exists():
            return handler.send(404, {'error': 'No completed context for that run'})
        try:
            packet = _read_json(run / 'context.json')
        except (OSError, ValueError):
            return handler.send(500, {'error': 'This saved inquiry could not be read; its files may be incomplete'})
        valid = {row.get('citation') for key in ('full_units', 'summaries', 'passages')
                 for row in (packet.get(key) or []) if isinstance(row, dict)}
        if citation not in valid:
            raise ValueError('That citation is not part of this inquiry')
        with verdict_lock:
            target = verdict_root / f'{run_id}.json'
            try:
                saved = _read_json(target)
                rows = saved['verdicts'] if isinstance(saved, dict) and isinstance(saved.get('verdicts'), dict) else {}
            except (OSError, ValueError, KeyError):
                rows = {}
            if verdict is None:
                rows.pop(citation, None)
            else:
                rows[citation] = {'verdict': verdict, 'updated': now()}
            try:
                atomic_json(target, {'schema': 'mori.verdicts.v1', 'run_id': run_id, 'verdicts': rows})
            except OSError:
                return handler.send(500, {'error': 'The check could not be saved in this room'})
        return handler.send(200, {'verdicts': read_verdicts(run_id)})

    Handler.save_verdict = save_verdict
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    port = server.server_address[1]
    origin = f'http://127.0.0.1:{port}'
    return server


def serve(store, config, port=8840, *, allow_uploads=True):
    server = create_server(store, config, port, allow_uploads=allow_uploads)
    print(f'Mori research room: http://127.0.0.1:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
