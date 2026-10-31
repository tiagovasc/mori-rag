"""Configured model choices and loopback request isolation; synthetic providers only."""
import json
import re
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

from mori.config import Config
from mori.providers import LLM
from mori.web import create_server


class ModelConfigurationTests(unittest.TestCase):
    def test_default_is_included_and_catalog_order_is_preserved(self):
        config = Config(llm_model='fixture-default', llm_models=['fixture-other', 'fixture-default']).validate()
        self.assertEqual(config.available_llm_models(), ['fixture-default', 'fixture-other'])
        self.assertEqual(config.llm_models, ['fixture-other', 'fixture-default'])
        self.assertEqual(Config().validate().available_llm_models(), [])

    def test_catalog_rejects_malformed_or_duplicate_ids(self):
        for catalog in ('fixture', None, [''], [' '], [' fixture'], ['fixture model'],
                        ['fixture\n'], ['fixture\x00'], [None], [1], [['fixture']],
                        ['fixture', 'fixture'], ['x' * 257], [f'fixture-{i}' for i in range(101)]):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                Config(llm_models=catalog).validate()

    def test_catalog_accepts_common_provider_id_formats(self):
        catalog = ['vendor/model:version', 'model_2.1-quantized', 'provider/model@revision']
        self.assertEqual(Config(llm_models=catalog).validate().available_llm_models(), catalog)


class ModelWebTests(unittest.TestCase):
    def start_server(self, config):
        store = Mock()
        store.stats.return_value = {'units': 0, 'chunks': 0}
        store.source_catalog.return_value = []
        self.config = config
        self.embedding_mock = self.enterContext(patch('mori.web.Embeddings'))
        self.http_mock = self.enterContext(patch('mori.providers.http_json', return_value={
            'choices': [{'message': {'content': 'Synthetic answer'}, 'finish_reason': 'stop'}]}))
        self.query_mock = self.enterContext(patch('mori.web.query', side_effect=lambda _store, question, active, **_kwargs: {
            'answer': LLM(active).complete('answer', question)}))
        self.plan_mock = self.enterContext(patch('mori.web.plan', return_value={'synthetic': True}))
        self.server = create_server(store, config, 0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=3)
        self.addCleanup(stop)
        self.origin = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        _, html = self.request('/')
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
            return response.code, response.read()

    def state(self):
        code, body = self.request('/api/state')
        self.assertEqual(code, 200)
        return json.loads(body)

    def wait_for_query(self):
        for _ in range(100):
            job = self.state()['job']
            if job['status'] != 'running':
                self.assertEqual(job['status'], 'complete', job)
                return
            time.sleep(.02)
        self.fail('Synthetic query did not finish')

    def test_selected_model_reaches_provider_and_next_request_keeps_server_default(self):
        self.start_server(Config(embedding='demo', llm='http', llm_model='fixture-default',
                                 llm_models=['fixture-other']))
        initial = self.state()
        self.assertEqual(initial['providers']['models'], ['fixture-default', 'fixture-other'])
        self.assertEqual(initial['providers']['model'], 'fixture-default')
        self.assertEqual(initial['settings']['llm_model'], 'fixture-default')
        for settings, expected in (({'llm_model': 'fixture-other'}, 'fixture-other'),
                                   ({'llm_model': 'fixture-default'}, 'fixture-default'),
                                   ({}, 'fixture-default')):
            self.assertEqual(self.request('/api/plan', {'settings': settings})[0], 200)
            self.assertEqual(self.plan_mock.call_args.args[1].llm_model, expected)
            self.assertEqual(self.request('/api/query', {'question': 'Synthetic question', 'answer': True,
                                                        'settings': settings})[0], 202)
            self.wait_for_query()
            self.assertEqual(self.query_mock.call_args.args[2].llm_model, expected)
            self.assertEqual(self.http_mock.call_args.args[1]['model'], expected)
            self.assertEqual(self.http_mock.call_args.args[0], self.config.llm_url)
            self.assertEqual(self.state()['providers'], initial['providers'])
            self.assertEqual(self.config.llm_model, 'fixture-default')
        self.embedding_mock.assert_called_once_with(self.config)

    def test_invalid_models_and_provider_overrides_are_rejected_before_provider_work(self):
        self.start_server(Config(embedding='demo', llm='http', llm_model='fixture-default',
                                 llm_models=['fixture-other']))
        invalid_settings = [{'llm_model': model} for model in
                            ('missing', '', 'fixture-other ', None, 1, True, [], {}, ['fixture-other'])]
        invalid_settings += [{'llm_models': ['untrusted']}, {'llm_url': 'https://untrusted.example'},
                             {'llm': 'command'}, {'llm_command': ['untrusted']}, {'llm_key_env': 'SECRET'},
                             {'embedding_url': 'https://untrusted.example'}]
        for settings in invalid_settings:
            for path in ('/api/plan', '/api/query', '/api/upload'):
                with self.subTest(settings=settings, path=path):
                    self.assertEqual(self.request(path, {'settings': settings, 'question': 'Synthetic'})[0], 400)
        self.embedding_mock.assert_not_called()
        self.query_mock.assert_not_called()
        self.http_mock.assert_not_called()
        self.plan_mock.assert_not_called()
        self.assertEqual(self.state()['job']['status'], 'idle')

    def test_empty_default_without_an_llm_preserves_existing_configuration(self):
        self.start_server(Config(embedding='demo'))
        self.assertEqual(self.state()['providers'], {'embedding': 'demo', 'llm': 'none', 'model': '', 'models': []})
        self.assertEqual(self.request('/api/plan', {'settings': {'llm_model': ''}})[0], 200)
        self.assertEqual(self.plan_mock.call_args.args[1].llm_model, '')
        self.assertEqual(self.request('/api/plan', {'settings': {'llm_model': 'unconfigured'}})[0], 400)
        self.http_mock.assert_not_called()

    def test_blank_provider_default_can_coexist_with_explicit_choices(self):
        self.start_server(Config(embedding='demo', llm='http', llm_models=['fixture-other']))
        self.assertEqual(self.state()['providers']['models'], ['fixture-other'])
        for settings, expected in (({}, ''), ({'llm_model': ''}, ''), ({'llm_model': 'fixture-other'}, 'fixture-other')):
            self.assertEqual(self.request('/api/plan', {'settings': settings})[0], 200)
            self.assertEqual(self.plan_mock.call_args.args[1].llm_model, expected)
        self.embedding_mock.assert_not_called()
        self.http_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
