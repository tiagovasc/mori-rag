"""Live rating board, stopping an inquiry, reading a unit, citation checks and fonts: synthetic, local HTTP only."""
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from mori.config import Config
from mori.ingest import ingest
from mori.live import LiveBoard, excerpt, level_of
from mori.pipeline import Cancelled, query
from mori.providers import Embeddings
from mori.store import Store
from mori.web import create_server

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = Config().rubric


class Scorer:
    """Deterministic ratings: unit 0 is most relevant, then 1, then 2."""
    def score(self, question, evidence, purpose):
        index = int(re.search(r'SENTINEL (\d)', evidence)[1]) if re.search(r'SENTINEL (\d)', evidence) else 2
        return {'score': [88.9, 55.6, 11.1][index], 'raw_score': [8.0, 5.0, 1.0][index], 'cached': False}


class Writer:
    def complete(self, task, prompt):
        return 'The synthetic source supports this answer. [P1]'


class BoardTests(unittest.TestCase):
    def test_levels_follow_search_order_and_strongest_keeps_the_best_four(self):
        board = LiveBoard('Why do wetlands store water?', RUBRIC, top=2)
        board.begin('rate_passages', 3)
        board.rated('rate_passages', 2, {'id': 'c', 'text': 'Walls fail. Wetlands store water slowly.',
                                         'rating': {'score': 77.8, 'raw_score': 7.0}, 'similarity': .2},
                    {'title': 'Chapter', 'locator': 'Handbook, chapter 1', 'source_id': 'book'})
        board.rated('rate_passages', 0, {'id': 'a', 'text': 'Unrelated.', 'rating': {'score': 0.0}, 'similarity': .9}, {})
        board.rated('rate_passages', 1, {'id': 'b', 'text': 'Some text.', 'rating': {'score': 44.4}, 'similarity': .5}, {})
        snap = board.snapshot()
        self.assertEqual(snap['passages'], {'total': 3, 'rated': 3, 'levels': [0, 4, 7]})
        self.assertEqual([row['id'] for row in snap['strongest']], ['c', 'b'])
        first = snap['strongest'][0]
        self.assertEqual((first['label'], first['locator'], first['level']), ('Very strong', 'Handbook, chapter 1', 7))
        self.assertEqual(first['excerpt'], 'Wetlands store water slowly.')
        self.assertEqual(snap['labels'][:2], ['Unrelated', 'Incidental'])

    def test_summaries_are_counted_and_bad_rows_never_raise(self):
        board = LiveBoard('q', RUBRIC)
        board.rated('rate_passages', 0, {'rating': {'score': 50}})        # before begin: ignored
        board.begin('rate_summaries', 2)
        board.rated('rate_summaries', 0, {'rating': {'score': 100}})
        board.rated('rate_summaries', 1, {'rating': 'damaged'})
        board.rated('rate_summaries', 1, None)
        board.note('shortlist', 3)
        board.note('anything-else', 'ignored')
        board.note('run', '../not-a-run')
        self.assertIsNone(board.run_id())
        board.note('run', '20261101T120000-abcdef0123')
        self.assertEqual(board.run_id(), '20261101T120000-abcdef0123')
        snap = board.snapshot()
        self.assertNotIn('run', snap)
        self.assertEqual(snap['summaries']['rated'], 2)
        self.assertEqual(snap['summaries']['counts'][9], 1)
        self.assertEqual(snap['shortlist'], 3)
        self.assertNotIn('anything-else', snap)
        snap['summaries']['counts'][0] = 99
        self.assertEqual(board.snapshot()['summaries']['counts'][0], 0)

    def test_excerpt_and_levels(self):
        self.assertEqual(level_of({'score': 77.8}, 10), 7)
        self.assertEqual(level_of({'raw_score': 6.6, 'score': 10}, 10), 7)
        self.assertIsNone(level_of({'score': float('nan')}, 10))
        long = 'Storage matters. ' + 'Wetland storage slows the flood peak across the catchment ' * 8
        self.assertLessEqual(len(excerpt(long, 'wetland storage flood')), 261)
        self.assertTrue(excerpt(long, 'wetland storage flood').endswith('…'))
        self.assertEqual(excerpt('', 'q'), '')


class Fixture(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='live-fixture', retrieval_k=3, jev_k=3,
                             final_passages=3, rerank_k=3, full_units=0, summaries=1, chunk_strategy='words',
                             chunk_words=40, chunk_overlap=0, min_relevance=0, summary_min_relevance=0,
                             workers=1).validate()
        self.embeddings = Embeddings(self.config)
        ingest(self.store, [{'id': f'u{index}', 'source_id': 'fixture', 'title': f'Unit {index}',
                             'text': f'First sentence. PRIVATE-SOURCE-SENTINEL {index} explains storage.',
                             'summary': f'Summary of unit {index}.'} for index in range(3)],
               self.config, self.embeddings)


class PipelineLiveTests(Fixture):
    def test_board_sees_every_rating_and_the_selected_counts(self):
        board = LiveBoard('storage', self.config.rubric)
        query(self.store, 'storage', self.config, embeddings=self.embeddings, scorer=Scorer(), live=board)
        snap = board.snapshot()
        self.assertEqual(snap['passages']['rated'], 3)
        self.assertEqual(sorted(snap['passages']['levels']), [1, 5, 8])
        self.assertEqual(snap['summaries']['rated'], 3)
        self.assertEqual(snap['shortlist'], 3)
        self.assertEqual(snap['evidence'], {'full_units': 0, 'summaries': 1, 'passages': 3})
        self.assertEqual(snap['strongest'][0]['title'], 'Unit 0')
        self.assertEqual(board.run_id(), next((Path(self.temp.name) / 'runs').iterdir()).name)

    def test_a_broken_board_never_changes_the_result(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError('display failure')
        result = query(self.store, 'storage', self.config, embeddings=self.embeddings, scorer=Scorer(), live=Broken())
        self.assertEqual(result['counts']['passages'], 3)

    def test_stop_during_rating_saves_a_stopped_status_and_progress(self):
        cancel, snapshots = threading.Event(), []
        class Stopping(Scorer):
            def score(inner, *args):
                cancel.set()
                return super().score(*args)
        with self.assertRaises(Cancelled):
            query(self.store, 'storage', self.config, embeddings=self.embeddings, scorer=Stopping(),
                  progress=snapshots.append, cancel=cancel)
        failed = [row for row in snapshots[-1]['steps'] if row['status'] == 'failed']
        self.assertEqual([row['id'] for row in failed], ['rate_passages'])
        self.assertIn('Stopped on request', failed[0]['message'])
        status = json.loads(next((Path(self.temp.name) / 'runs').glob('*/status.json')).read_text())
        self.assertEqual(status['status'], 'stopped')

    def test_unit_text_returns_offsets_without_vectors(self):
        unit = self.store.unit_text('u1')
        self.assertEqual(unit['title'], 'Unit 1')
        self.assertTrue(unit['passages'])
        self.assertNotIn('summary_vector', unit)
        first = unit['passages'][0]
        self.assertEqual(set(first), {'id', 'seq', 'start', 'end'})
        self.assertIsNone(self.store.unit_text('missing'))
        self.assertIsNone(self.store.unit_text(''))


class WebLiveTests(Fixture):
    def start(self, scorer):
        self.enterContext(patch('mori.web.Embeddings', return_value=self.embeddings))
        self.enterContext(patch('mori.pipeline.Jev', return_value=scorer))
        self.enterContext(patch('mori.pipeline.LLM', return_value=Writer()))
        self.config = replace(self.config, llm='command', llm_command=['synthetic-answer-model']).validate()
        self.server = create_server(self.store, self.config, 0)
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()
        def stop():
            self.server.shutdown()
            self.server.server_close()
            worker.join(3)
        self.addCleanup(stop)
        self.origin = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', self.request('/').decode())[1]

    def request(self, path, body=None, raw=False):
        headers = {'Origin': self.origin, 'X-Mori-Token': getattr(self, 'token', ''), 'Content-Type': 'application/json'}
        request = urllib.request.Request(self.origin + path, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        with self.opener.open(request, timeout=3) as response:
            return response if raw else response.read()

    def error(self, path, body=None):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(path, body)
        return caught.exception.code, json.loads(caught.exception.read())

    def wait(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = json.loads(self.request('/api/live'))['job']
            if job['status'] != 'running':
                return job
            time.sleep(.01)
        self.fail('Synthetic query did not finish')

    def test_live_board_stop_and_features(self):
        entered, release = threading.Event(), threading.Event()
        class Blocked(Scorer):
            def score(inner, *args):
                entered.set()
                release.wait(5)
                return super().score(*args)
        self.start(Blocked())
        state = json.loads(self.request('/api/state'))
        self.assertEqual(set(state['features']), {'live', 'cancel', 'fonts', 'verdicts', 'unit_text'})
        self.assertEqual(self.error('/api/cancel', {})[0], 409)
        self.request('/api/query', {'question': 'storage', 'sources': ['fixture']})
        self.assertTrue(entered.wait(2))
        running = json.loads(self.request('/api/live'))
        self.assertEqual(running['job']['request']['question'], 'storage')
        self.assertEqual(running['job']['request']['settings']['retrieval_k'], 3)
        self.assertEqual(running['live']['passages']['total'], 3)
        # The running inquiry's folder has no status yet; the list reports it as running, not interrupted.
        self.assertEqual(json.loads(self.request('/api/runs'))[0]['status'], 'running')
        self.assertEqual(json.loads(self.request('/api/cancel', {})), {'status': 'stopping'})
        self.assertTrue(json.loads(self.request('/api/state'))['job']['stopping'])
        release.set()
        job = self.wait()
        self.assertEqual(job['status'], 'stopped')
        self.assertIn('Stopped on request', job['error'])
        runs = json.loads(self.request('/api/runs'))
        self.assertEqual(runs[0]['status'], 'stopped')

    def test_unit_reader_verdicts_and_summaries(self):
        self.start(Scorer())
        unit = json.loads(self.request('/api/unit?id=u2'))
        self.assertIn('PRIVATE-SOURCE-SENTINEL 2', unit['text'])
        self.assertEqual(unit['collection'], 'fixture')
        self.assertEqual(self.error('/api/unit?id=missing')[0], 404)
        self.assertEqual(self.error('/api/unit?id=a&id=b')[0], 400)
        self.request('/api/query', {'question': 'storage', 'answer': True, 'sources': ['fixture']})
        job = self.wait()
        self.assertEqual(job['status'], 'complete', job)
        run_id = job['result']['run_id']
        run_dir = Path(self.temp.name) / 'runs' / run_id
        before = sorted(path.name for path in run_dir.iterdir())
        path = f'/api/runs/{run_id}/verdicts'
        self.assertEqual(self.error(path, {'citation': 'P99', 'verdict': 'yes'})[0], 400)
        self.assertEqual(self.error(path, {'citation': '../P1', 'verdict': 'yes'})[0], 400)
        self.assertEqual(self.error(path, {'citation': 'P1', 'verdict': 'maybe'})[0], 400)
        self.assertEqual(json.loads(self.request(path, {'citation': 'P1', 'verdict': 'partly'})), {'verdicts': {'P1': 'partly'}})
        saved = json.loads(self.request(f'/api/runs/{run_id}'))
        self.assertEqual(saved['verdicts'], {'P1': 'partly'})
        summary = json.loads(self.request('/api/runs'))[0]
        self.assertEqual((summary['citations'], summary['cited'], summary['checked']), (1, 1, 1))
        self.assertEqual(summary['sources'], ['fixture'])
        self.assertEqual(summary['settings']['retrieval_k'], 3)
        self.assertEqual(json.loads(self.request(path, {'citation': 'P1', 'verdict': None})), {'verdicts': {}})
        self.assertEqual(sorted(path.name for path in run_dir.iterdir()), before)
        self.assertTrue((Path(self.temp.name) / 'verdicts' / f'{run_id}.json').is_file())

    def test_fonts_are_allowlisted(self):
        self.start(Scorer())
        names = [name for name in ('literata-latin.woff2', 'source-sans-3-latin.woff2')
                 if (ROOT / 'mori' / 'static' / 'fonts' / name).is_file()]
        self.assertEqual(len(names), 2, 'bundled fonts are missing')
        response = self.request('/fonts/' + names[0], raw=True)
        self.assertEqual(response.headers['Content-Type'], 'font/woff2')
        self.assertIn('max-age', response.headers['Cache-Control'])
        for path in ('/fonts/../web.py', '/fonts/OFL.txt', '/fonts/'):
            self.assertEqual(self.error(path)[0], 404)


if __name__ == '__main__':
    unittest.main()
