"""Actual progress boundaries with synthetic sources and local HTTP only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request

from mori.config import Config
from mori.ingest import ingest
from mori.pipeline import query
from mori.progress import Progress, QUERY_STEPS
from mori.providers import Embeddings
from mori.store import Store, atomic_json
from mori.web import create_server


ROOT = Path(__file__).resolve().parents[1]


def steps(snapshot):
    return {row['id']: row for row in snapshot['steps']}


class Scorer:
    def score(self, question, evidence, purpose):
        return {'score': 80, 'cached': True}


class Writer:
    def json(self, task, prompt):
        payload = json.loads(prompt[prompt.index('{"question"'):])
        return {'ids': [row['id'] for row in reversed(payload['passages'])]}

    def complete(self, task, prompt):
        return 'The synthetic source supports this answer. [P1]'


class Fixture(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='progress-fixture',
                             retrieval_k=3, jev_k=3, final_passages=3, rerank_k=3,
                             full_units=0, summaries=0, chunk_words=40, chunk_overlap=0,
                             min_relevance=0, summary_min_relevance=0, workers=2).validate()
        self.embeddings = Embeddings(self.config)
        ingest(self.store, [{'id': str(index), 'source_id': 'fixture', 'title': 'Synthetic unit',
                            'text': f'PRIVATE-SOURCE-SENTINEL {index}',
                            'summary': f'PRIVATE-SUMMARY-SENTINEL {index}'}
                           for index in range(3)], self.config, self.embeddings)

    def run_query(self, **kwargs):
        return query(self.store, 'PRIVATE-QUESTION-SENTINEL', self.config,
                     embeddings=self.embeddings, **kwargs)


class PipelineProgressTests(Fixture):
    def test_evidence_only_skips_optional_steps_and_exposes_no_source_or_question(self):
        snapshots = []
        result = self.run_query(scorer=Scorer(), progress=snapshots.append)
        final = snapshots[-1]
        self.assertEqual(final['percent'], 100)
        self.assertTrue(final['finished_at'])
        self.assertIsNone(final['current_stage'])
        self.assertEqual([row['id'] for row in final['steps']], [key for key, _ in QUERY_STEPS])
        for stage in ('rate_summaries', 'rerank', 'answer', 'validate'):
            self.assertEqual(steps(final)[stage]['status'], 'skipped')
        self.assertEqual(steps(final)['rate_passages']['completed'], result['counts']['rated_passages'])
        self.assertEqual(steps(final)['rate_passages']['cached'], 3)
        self.assertTrue(all(snapshot['percent'] < 100 for snapshot in snapshots[:-1]))
        self.assertEqual([s['percent'] for s in snapshots], sorted(s['percent'] for s in snapshots))
        serialized = json.dumps(snapshots)
        for sentinel in ('PRIVATE-SOURCE', 'PRIVATE-SUMMARY', 'PRIVATE-QUESTION'):
            self.assertNotIn(sentinel, serialized)

    def test_answer_summary_and_rerank_stages_run_when_requested(self):
        self.config.summaries = 1
        self.config.rerank_enabled = True
        snapshots = []
        self.run_query(scorer=Scorer(), llm=Writer(), answer=True, progress=snapshots.append)
        self.assertTrue(all(row['status'] == 'complete' for row in snapshots[-1]['steps']))
        self.assertEqual(steps(snapshots[-1])['rate_summaries']['completed'], 3)
        active = [s['current_stage'] for s in snapshots if s['current_stage']]
        self.assertEqual(list(dict.fromkeys(active)), [key for key, _ in QUERY_STEPS])

    def test_counts_follow_completed_futures_while_first_submission_is_blocked(self):
        first_started, release_first, increment_seen = threading.Event(), threading.Event(), threading.Event()
        call_lock = threading.Lock()
        call_count = 0
        snapshots, outcomes = [], []

        class OutOfOrder:
            def score(inner, question, evidence, purpose):
                nonlocal call_count
                with call_lock:
                    call_count += 1
                    index = call_count
                if index == 1:
                    first_started.set()
                    if not release_first.wait(5):
                        raise RuntimeError('Test failed to release the first rating')
                return {'score': 80, 'cached': index != 1}

        def observe(snapshot):
            snapshots.append(snapshot)
            if steps(snapshot)['rate_passages']['completed'] in (1, 2):
                increment_seen.set()

        def work():
            try:
                outcomes.append(self.run_query(scorer=OutOfOrder(), progress=observe))
            except Exception as error:
                outcomes.append(error)

        worker = threading.Thread(target=work)
        worker.start()
        try:
            self.assertTrue(first_started.wait(2))
            self.assertTrue(increment_seen.wait(2), 'A later completed rating was hidden behind the first future')
            partial = steps(snapshots[-1])['rate_passages']
            self.assertEqual(partial['status'], 'running')
            self.assertLess(partial['completed'], partial['total'])
            self.assertEqual(partial['cached'], partial['completed'])
            self.assertTrue(worker.is_alive())
        finally:
            release_first.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcomes[0], dict, outcomes)
        self.assertEqual(steps(snapshots[-1])['rate_passages']['cached'], 2)

    def test_rejected_answer_fails_validation_without_fake_completion(self):
        class Rejected(Writer):
            def complete(self, *_):
                return 'Unsupported generated output. [P999]'
        snapshots = []
        with self.assertRaises(ValueError):
            self.run_query(scorer=Scorer(), llm=Rejected(), answer=True, progress=snapshots.append)
        final = snapshots[-1]
        self.assertEqual(final['current_stage'], 'validate')
        self.assertEqual(steps(final)['answer']['status'], 'complete')
        self.assertEqual(steps(final)['validate']['status'], 'failed')
        self.assertEqual(steps(final)['save']['status'], 'pending')
        self.assertLess(final['percent'], 100)
        self.assertTrue(final['finished_at'])

    def test_summary_threshold_failure_does_not_blame_disabled_reranker(self):
        self.config.full_units = 1
        self.config.summary_min_relevance = 99
        snapshots = []
        with self.assertRaisesRegex(ValueError, 'Not enough relevant full units'):
            self.run_query(scorer=Scorer(), progress=snapshots.append)
        self.assertEqual(snapshots[-1]['current_stage'], 'rate_summaries')
        self.assertEqual(steps(snapshots[-1])['rate_summaries']['status'], 'failed')
        self.assertNotEqual(steps(snapshots[-1])['rerank']['status'], 'failed')

    def test_save_failure_never_reports_one_hundred_percent(self):
        snapshots = []
        def write(path, value):
            if path.name == 'status.json' and value['status'] == 'complete':
                raise OSError('Synthetic save failure')
            return atomic_json(path, value)
        with patch('mori.pipeline.atomic_json', side_effect=write), self.assertRaises(OSError):
            self.run_query(scorer=Scorer(), progress=snapshots.append)
        self.assertEqual(steps(snapshots[-1])['save']['status'], 'failed')
        self.assertTrue(all(s['percent'] < 100 for s in snapshots))

    def test_ratings_write_failure_remains_on_active_ratings_stage(self):
        snapshots = []
        def write(path, value):
            if path.name == 'passage_ratings.json':
                raise OSError('Synthetic ratings save failure')
            return atomic_json(path, value)
        with patch('mori.pipeline.atomic_json', side_effect=write), self.assertRaises(OSError):
            self.run_query(scorer=Scorer(), progress=snapshots.append)
        self.assertEqual(steps(snapshots[-1])['rate_passages']['status'], 'failed')
        self.assertNotEqual(steps(snapshots[-1])['rerank']['status'], 'failed')

    def test_parallel_failure_is_visible_before_other_inflight_calls_return(self):
        blocked, release, failure_seen = threading.Event(), threading.Event(), threading.Event()
        counter, call_lock = 0, threading.Lock()
        outcomes, snapshots = [], []
        class Failing:
            def score(inner, *_):
                nonlocal counter
                with call_lock:
                    counter += 1
                    current = counter
                if current == 1:
                    blocked.set()
                    release.wait(5)
                    return {'score': 80}
                blocked.wait(2)
                raise RuntimeError('Synthetic scorer failure')
        def observe(snapshot):
            snapshots.append(snapshot)
            if steps(snapshot)['rate_passages']['status'] == 'failed':
                failure_seen.set()
        def work():
            try:
                self.run_query(scorer=Failing(), progress=observe)
            except RuntimeError as error:
                outcomes.append(error)
        worker = threading.Thread(target=work)
        worker.start()
        try:
            self.assertTrue(failure_seen.wait(2))
            self.assertTrue(worker.is_alive())
            frozen = deepcopy(snapshots[-1])
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(snapshots[-1], frozen)
        self.assertEqual(steps(frozen)['rate_passages']['completed'], 0)


class ProgressStateTests(unittest.TestCase):
    def test_thread_safe_counts_snapshot_isolation_and_failure_freeze(self):
        snapshots = []
        tracker = Progress(snapshots.append, [('work', 'Synthetic work')])
        tracker.start('work', total=400)
        def count():
            for _ in range(100):
                tracker.advance('work', cached=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: count(), range(4)))
        self.assertEqual(steps(tracker.snapshot())['work']['completed'], 400)
        self.assertEqual(steps(tracker.snapshot())['work']['cached'], 400)
        self.assertEqual([steps(s)['work']['completed'] for s in snapshots], list(range(401)))
        snapshots[-1]['steps'][0]['completed'] = -999
        self.assertEqual(steps(tracker.snapshot())['work']['completed'], 400)
        tracker.fail()
        failed = tracker.snapshot()
        tracker.complete('work')
        tracker.finish()
        self.assertEqual(tracker.snapshot(), failed)
        self.assertLess(failed['percent'], 100)


class WebProgressTests(Fixture):
    def start_server(self, embedding_factory=None):
        self.enterContext(patch('mori.web.Embeddings', side_effect=embedding_factory or (lambda _: self.embeddings)))
        self.enterContext(patch('mori.pipeline.Jev', return_value=Scorer()))
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
        html = self.request('/').decode()
        self.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', html)[1]

    def request(self, path, body=None):
        headers = {'Origin': self.origin, 'X-Mori-Token': getattr(self, 'token', ''), 'Content-Type': 'application/json'}
        request = urllib.request.Request(self.origin + path, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        with self.opener.open(request, timeout=2) as response:
            return response.read()

    def job(self):
        return json.loads(self.request('/api/state'))['job']

    def wait_for_finish(self):
        deadline = time.monotonic() + 5
        observed = []
        while time.monotonic() < deadline:
            job = self.job()
            observed.append(job)
            if job['status'] != 'running':
                self.assertTrue(all(j['progress']['percent'] < 100 for j in observed[:-1]))
                return job
            time.sleep(.01)
        self.fail('Synthetic query did not finish')

    def test_provider_startup_is_visible_and_new_jobs_reset_completed_progress(self):
        entered, release = threading.Event(), threading.Event()
        def initialize(_):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test failed to release provider initialization')
            return self.embeddings
        self.start_server(initialize)
        try:
            self.request('/api/query', {'question': 'Synthetic query'})
            self.assertTrue(entered.wait(2))
            initial = self.job()
            self.assertEqual(initial['kind'], 'query')
            self.assertEqual(initial['progress']['current_stage'], 'prepare')
            self.assertEqual(initial['progress']['percent'], 0)
            self.assertIsNone(initial['progress']['finished_at'])
        finally:
            release.set()
        complete = self.wait_for_finish()
        self.assertEqual(complete['status'], 'complete')
        self.assertEqual(complete['progress']['percent'], 100)
        self.assertIn('run_id', complete['result'])
        self.request('/api/query', {'question': 'Another synthetic query'})
        again = self.wait_for_finish()
        self.assertEqual(again['status'], 'complete')
        self.assertNotEqual(initial['progress']['started_at'], again['progress']['started_at'])
        self.assertNotEqual(complete['result']['run_id'], again['result']['run_id'])

    def test_state_polls_remain_responsive_during_blocked_scoring(self):
        entered, release = threading.Event(), threading.Event()
        self.start_server()
        class Blocked(Scorer):
            def score(inner, *args):
                entered.set()
                release.wait(5)
                return super().score(*args)
        with patch('mori.pipeline.Jev', return_value=Blocked()):
            try:
                self.request('/api/query', {'question': 'Synthetic query'})
                self.assertTrue(entered.wait(2))
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = list(pool.map(lambda _: self.job(), range(8)))
                for job in jobs:
                    self.assertEqual(job['status'], 'running')
                    stage = steps(job['progress'])['rate_passages']
                    self.assertEqual((stage['completed'], stage['total']), (0, 3))
                    self.assertEqual(job['progress']['current_stage'], 'rate_passages')
            finally:
                release.set()
            finished = self.wait_for_finish()
            self.assertEqual(finished['status'], 'complete')
            self.assertEqual(steps(finished['progress'])['rate_passages']['completed'], 3)

    def test_provider_failure_is_attributed_to_prepare_without_sensitive_details_in_progress(self):
        def fail(_):
            raise RuntimeError('SYNTHETIC-PROVIDER-FAILURE')
        self.start_server(fail)
        self.request('/api/query', {'question': 'Synthetic query'})
        job = self.wait_for_finish()
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(steps(job['progress'])['prepare']['status'], 'failed')
        self.assertLess(job['progress']['percent'], 100)
        self.assertNotIn('SYNTHETIC-PROVIDER-FAILURE', json.dumps(job['progress']))

    def test_upload_reports_honest_read_and_import_stages(self):
        self.start_server()
        entered, release = threading.Event(), threading.Event()
        real_ingest = ingest
        def blocked_import(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test failed to release import')
            return real_ingest(*args)
        units = [{'id': 'uploaded', 'source_id': 'upload-fixture', 'title': 'Synthetic import',
                  'text': 'A new uploaded source.', 'summary': 'Complete synthetic summary.'}]
        with patch('mori.web.ingest', side_effect=blocked_import):
            try:
                self.request('/api/upload', {'filename': 'synthetic.jsonl',
                                             'data': base64.b64encode(json.dumps(units[0]).encode()).decode()})
                self.assertTrue(entered.wait(2))
                job = self.job()
                self.assertEqual(job['kind'], 'upload')
                self.assertEqual(job['progress']['current_stage'], 'import')
                self.assertEqual(steps(job['progress'])['read']['status'], 'complete')
                self.assertIsNone(steps(job['progress'])['import']['total'])
                self.assertLess(job['progress']['percent'], 100)
            finally:
                release.set()
            final = self.wait_for_finish()
        self.assertEqual(final['status'], 'complete')
        self.assertEqual(final['result']['added_units'], 1)
        self.assertEqual(final['progress']['percent'], 100)
        self.assertEqual([row['id'] for row in final['progress']['steps']], ['prepare', 'read', 'import', 'save'])


if __name__ == '__main__':
    unittest.main()
