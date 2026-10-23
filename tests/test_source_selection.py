"""Source-selection boundaries, using only synthetic fixtures and loopback HTTP."""
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.parse
import urllib.request

from mori.config import Config
from mori.ingest import ingest
from mori.pipeline import plan, query
from mori.providers import Embeddings
from mori.store import Store, atomic_json
from mori.web import create_server


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "collection / α?&'"
OTHER = 'different-source'


class Fixture(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='source-selection-fixture',
                             chunk_words=8, chunk_overlap=2, retrieval_k=30, jev_k=20,
                             final_passages=3, full_units=1, summaries=1, workers=1,
                             min_relevance=0, summary_min_relevance=0,
                             rating_neighbors=1, answer_neighbors=2).validate()
        self.embeddings = Embeddings(self.config)
        self.originals = [
            {'id': uid, 'source_id': source, 'title': title,
             'text': ' '.join(f'{marker}-{n}' for n in range(28)),
             'summary': f'{marker} complete synthetic summary.', 'metadata': {'fixture': marker}}
            for uid, source, title, marker in (
                ('unit-a', SOURCE, 'Selected volume', 'selected-alpha'),
                ('unit-b', SOURCE, 'Selected volume', 'selected-beta'),
                ('unit-c', OTHER, 'Other volume', 'EXCLUDED-CONTENT'))]
        ingest(self.store, self.originals, self.config, self.embeddings)


class SourceStoreTests(Fixture):
    def test_catalog_counts_without_loading_text_or_vectors(self):
        with patch.object(self.store, 'units', side_effect=AssertionError('Loaded units')), \
                patch.object(self.store, 'chunks', side_effect=AssertionError('Loaded vectors')):
            catalog = {row['id']: row for row in self.store.source_catalog()}
        self.assertEqual(catalog[SOURCE], {'id': SOURCE, 'label': 'Selected volume', 'units': 2, 'chunks': 10})
        self.assertEqual(catalog[OTHER]['chunks'], 5)
        self.assertEqual(sum(row['units'] for row in catalog.values()), self.store.stats()['units'])

    def test_filtered_view_is_readonly_and_shares_fingerprint_cache_and_runs(self):
        selected = self.store.select_sources([SOURCE])
        self.assertEqual({u['id'] for u in selected.units()}, {'unit-a', 'unit-b'})
        self.assertEqual({r['unit_id'] for r in selected.chunks()}, {'unit-a', 'unit-b'})
        self.assertEqual(selected.stats(), {'units': 2, 'chunks': 10, 'embedding': self.store.fingerprint()})
        self.assertEqual(selected.root, self.store.root)
        self.assertEqual(selected.fingerprint(), self.store.fingerprint())
        self.assertEqual(selected.run_roots(), [self.store.root])
        selected.cache_put('synthetic-rating', {'score': 80})
        self.assertEqual(self.store.cache_get('synthetic-rating'), {'score': 80})
        self.assertEqual(selected.cache_get('synthetic-rating'), {'score': 80})
        self.assertEqual(selected.source_catalog(), selected.selection)
        with self.assertRaisesRegex(ValueError, 'read-only'):
            selected.add([], {}, {})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            selected.select_sources([OTHER])
        self.assertEqual(selected.select_sources([SOURCE]).stats(), selected.stats())

    def test_invalid_selection_fails_closed_and_ids_remain_exact(self):
        for selection in ([], None, SOURCE, {}, [1], [None], [''], ['  '],
                          [SOURCE, SOURCE], ['unknown'], [SOURCE, 'unknown'], [SOURCE + ' ']):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.store.select_sources(selection)
        selected = self.store.select_sources([OTHER, SOURCE])
        self.assertEqual([r['id'] for r in selected.selection], [OTHER, SOURCE])
        self.assertEqual(selected.stats(), self.store.stats())

    def test_selection_bounds_every_pipeline_evidence_stage_and_budget(self):
        selected = self.store.select_sources([SOURCE])
        scorer = Mock()
        scorer.score.return_value = {'score': 80, 'label': 'synthetic', 'cached': False}
        result = query(selected, 'Explain the synthetic material.', self.config,
                       embeddings=self.embeddings, scorer=scorer)
        run = Path(result['directory'])
        packet = json.loads((run / 'context.json').read_text(encoding='utf-8'))
        request = json.loads((run / 'request.json').read_text(encoding='utf-8'))
        expected = [{'id': SOURCE, 'label': 'Selected volume'}]
        self.assertEqual(packet['sources'], expected)
        self.assertEqual(request['sources'], expected)
        self.assertEqual((len(packet['full_units']), len(packet['summaries']), len(packet['passages'])), (1, 1, 3))
        self.assertEqual({u['source_id'] for key in ('full_units', 'summaries') for u in packet[key]}, {SOURCE})
        chunks_by_id = {row['id']: row for row in selected.chunks()}
        for passage in packet['passages']:
            self.assertEqual(passage['unit']['source_id'], SOURCE)
            self.assertTrue(passage['neighbors'])
            for neighbor in passage['neighbors']:
                self.assertEqual(chunks_by_id[neighbor['id']]['unit_id'], passage['unit']['id'])
        for item in run.iterdir():
            self.assertNotIn('EXCLUDED-CONTENT', item.read_text(encoding='utf-8'), item.name)
        for call in scorer.score.call_args_list:
            self.assertNotIn('EXCLUDED-CONTENT', call.args[1])
        self.assertEqual(plan(selected, self.config)['maximum_rating_decisions'], 12)
        self.assertEqual(plan(self.store, self.config)['maximum_rating_decisions'], 18)

    def test_omitting_selection_preserves_all_source_legacy_packets(self):
        scorer = Mock()
        scorer.score.return_value = {'score': 80, 'label': 'synthetic', 'cached': False}
        result = query(self.store, 'Synthetic question', self.config, embeddings=self.embeddings, scorer=scorer)
        run = Path(result['directory'])
        for name in ('context.json', 'request.json'):
            self.assertNotIn('sources', json.loads((run / name).read_text(encoding='utf-8')))
        self.assertEqual(result['counts']['rated_summaries'], 3)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class SourceWebTests(Fixture):
    def setUp(self):
        super().setUp()
        self.embedding_mock = self.enterContext(patch('mori.web.Embeddings', return_value=self.embeddings))
        self.query_mock = self.enterContext(patch('mori.web.query', return_value={'synthetic': True}))
        self.server = create_server(self.store, self.config, 0, allow_uploads=False)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=3)
        self.addCleanup(stop)
        self.origin = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        _, html, _ = self.request('/')
        self.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', html.decode())[1]

    def request(self, path, body=None):
        headers = {'Origin': self.origin, 'X-Mori-Token': getattr(self, 'token', ''), 'Content-Type': 'application/json'}
        request = urllib.request.Request(self.origin + path, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        try:
            response = self.opener.open(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, response.read(), response.headers

    def test_state_selected_budget_and_invalid_sources_never_initialize_providers(self):
        code, body, _ = self.request('/api/state')
        self.assertEqual(code, 200)
        state = json.loads(body)
        self.assertEqual(state['sources'], self.store.source_catalog())
        self.assertFalse(state['allow_uploads'])
        code, body, _ = self.request('/api/plan', {'sources': [SOURCE]})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)['maximum_rating_decisions'], 12)
        self.assertEqual(json.loads(self.request('/api/plan', {})[1])['maximum_rating_decisions'], 18)
        for sources in ([], ['missing'], [SOURCE, SOURCE], [1], None, SOURCE):
            for path in ('/api/plan', '/api/query'):
                self.assertEqual(self.request(path, {'sources': sources, 'question': 'Synthetic question'})[0], 400)
        self.embedding_mock.assert_not_called()
        self.query_mock.assert_not_called()

    def test_query_receives_filtered_view_and_invalid_selection_precedes_busy_lock(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def work(store, *_args, **_kwargs):
            started.set()
            release.wait(timeout=3)
            finished.set()
            return {'units': store.stats()['units']}
        self.query_mock.side_effect = work
        try:
            self.assertEqual(self.request('/api/query', {'sources': [SOURCE], 'question': 'Synthetic question'})[0], 202)
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(self.request('/api/query', {'sources': [], 'question': 'Invalid'})[0], 400)
            self.assertEqual(self.request('/api/query', {'sources': [OTHER], 'question': 'Busy'})[0], 409)
            selected = self.query_mock.call_args.args[0]
            self.assertEqual({unit['source_id'] for unit in selected.units()}, {SOURCE})
            self.assertEqual(selected.selection[0]['id'], SOURCE)
            self.embedding_mock.assert_called_once()
        finally:
            release.set()
            self.assertTrue(finished.wait(timeout=2))

    def test_disabled_uploads_reject_before_validation_and_filesystem_or_provider_work(self):
        self.assertEqual(self.request('/api/upload', {})[0], 403)
        self.assertEqual(self.request('/api/upload', {'settings': {'invalid': True}})[0], 403)
        self.assertFalse((self.store.root / 'uploads').exists())
        self.embedding_mock.assert_not_called()

    def test_source_open_redirect_is_relative_encoded_and_known_only(self):
        path = '/open?' + urllib.parse.urlencode({'collection': SOURCE})
        code, _, headers = self.request(path)
        self.assertEqual(code, 302)
        self.assertEqual(headers['Location'], '/?' + urllib.parse.urlencode({'sources': SOURCE}))
        self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(headers['Location']).query), {'sources': [SOURCE]})
        for path in ('/open', '/open?collection=', '/open?collection=missing',
                     '/open?collection=different-source&collection=different-source',
                     '/open?collection=https%3A%2F%2Funtrusted.example'):
            self.assertEqual(self.request(path)[0], 404)

    def test_history_aggregates_trusted_roots_and_keeps_own_copy_for_duplicate_ids(self):
        legacy = self.store.root / 'synthetic-legacy'
        own_id = '20260101T010101-0123456789'
        legacy_id = '20260102T010101-abcdef0123'
        for root, run_id, label in ((self.store.root, own_id, 'own'), (legacy, own_id, 'duplicate'), (legacy, legacy_id, 'legacy')):
            run = root / 'runs' / run_id
            run.mkdir(parents=True)
            atomic_json(run / 'request.json', {'question': label, 'created': 'synthetic'})
            atomic_json(run / 'status.json', {'status': 'complete'})
            atomic_json(run / 'context.json', {'fixture': label})
            (run / 'answer_prompt.txt').write_text(label, encoding='utf-8')
        self.store.run_roots = lambda: [self.store.root, legacy, legacy]
        code, body, _ = self.request('/api/runs')
        self.assertEqual(code, 200)
        self.assertEqual([(row['id'], row['question']) for row in json.loads(body)], [(legacy_id, 'legacy'), (own_id, 'own')])
        for run_id, label in ((own_id, 'own'), (legacy_id, 'legacy')):
            code, body, _ = self.request('/api/runs/' + run_id)
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)['context']['fixture'], label)
        self.assertEqual(self.request('/api/runs/../../outside')[0], 404)
        self.assertEqual(self.request('/api/runs/20260103T010101-0123456789')[0], 404)


if __name__ == '__main__':
    unittest.main()
