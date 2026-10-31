"""Conservative prompt budgets and per-request controls; no external providers."""
import json
import re
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from unittest.mock import Mock, patch

from mori.budget import COUNTING_NOTE, ContextBudget
from mori.config import Config
from mori.web import create_server


class ContextBudgetTests(unittest.TestCase):
    def test_complete_unicode_prompt_is_counted_without_a_word_ratio(self):
        prompt = 'Instructions\nQuestion?\n[U1] Café 中文 🐕\n'
        budget = ContextBudget(Config(evidence_mode='context'))
        measured = budget.measure(prompt)
        self.assertEqual(budget.count(prompt), len(prompt.encode('utf-8')))
        self.assertGreater(budget.count(prompt), len(prompt))
        self.assertEqual(measured['mode'], 'context')
        self.assertEqual(measured['limit_tokens'], 100000)
        self.assertEqual(measured['output_reserve_tokens'], 4096)
        self.assertEqual(measured['overhead_reserve_tokens'], 1024)
        self.assertEqual(measured['input_budget_tokens'], 94880)
        self.assertEqual(measured['remaining_tokens'], 94880 - budget.count(prompt))
        self.assertEqual(measured['counting_method'], 'utf8_bytes')
        self.assertEqual(measured['counting_note'], COUNTING_NOTE)
        self.assertIn('not an exact model token count', measured['counting_note'])

    def test_reserves_are_removed_before_the_exact_fit_boundary(self):
        budget = ContextBudget(Config(evidence_mode='context', context_window_tokens=100,
                                      llm_max_output_tokens=10, context_overhead_tokens=5))
        self.assertTrue(budget.fits('a' * 85))
        self.assertEqual(budget.measure('a' * 85)['remaining_tokens'], 0)
        self.assertFalse(budget.fits('a' * 86))
        self.assertEqual(budget.measure('a' * 86)['remaining_tokens'], -1)
        self.assertTrue(budget.fits('é' * 42))
        self.assertFalse(budget.fits('é' * 43))

    def test_separate_input_character_safety_limit_remains_enforced(self):
        budget = ContextBudget(Config(evidence_mode='context', llm_max_input_chars=10))
        self.assertTrue(budget.fits('a' * 10))
        self.assertFalse(budget.fits('a' * 11))
        self.assertGreater(budget.measure('a' * 11)['remaining_tokens'], 0)

    def test_budget_snapshots_configuration_and_does_not_rewrite_the_prompt(self):
        config = Config(evidence_mode='context')
        budget = ContextBudget(config)
        config.context_window_tokens = 200000
        config.llm_max_output_tokens = 10
        config.context_overhead_tokens = 0
        prompt = '<|special-token-looking-text|>\nExact original source'
        self.assertEqual(budget.measure(prompt)['input_budget_tokens'], 94880)
        self.assertEqual(budget.count(prompt), len(prompt.encode()))
        self.assertEqual(prompt, '<|special-token-looking-text|>\nExact original source')

    def test_non_string_prompt_is_rejected(self):
        budget = ContextBudget(Config())
        for prompt in (None, True, 1, [], {}, b'bytes'):
            with self.subTest(prompt=prompt), self.assertRaises(TypeError):
                budget.measure(prompt)


class ContextConfigurationTests(unittest.TestCase):
    def test_fixed_mode_remains_default_and_retains_count_ordering(self):
        self.assertEqual(Config().validate().evidence_mode, 'fixed')
        for rerank in (False, True):
            with self.subTest(rerank=rerank), self.assertRaisesRegex(ValueError, 'Passage counts'):
                Config(retrieval_k=10, jev_k=5, final_passages=6, rerank_k=6,
                       rerank_enabled=rerank).validate()

    def test_context_mode_ignores_fixed_caps_but_keeps_retrieval_ordering(self):
        for rerank in (False, True):
            Config(evidence_mode='context', retrieval_k=10, jev_k=5,
                   final_passages=20, rerank_k=20, rerank_enabled=rerank).validate()
        with self.assertRaisesRegex(ValueError, 'Passage counts'):
            Config(evidence_mode='context', retrieval_k=10, jev_k=11).validate()

    def test_invalid_modes_tokenizers_and_token_values_are_rejected(self):
        cases = [('evidence_mode', value) for value in ('', 'auto', None, True)]
        cases += [('context_tokenizer', value) for value in ('', 'unknown', 'o200k_base', None, True)]
        cases += [('context_window_tokens', value) for value in (0, -1, True, 1.5, '100000', None, 2000001)]
        cases += [('context_overhead_tokens', value) for value in (-1, True, 1.5, '1024', None)]
        cases += [('llm_max_output_tokens', value) for value in (0, -1, True, 1.5, '4096', None)]
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(Config(), **{field: value}).validate()
        Config(context_window_tokens=2000000, context_overhead_tokens=0).validate()

    def test_context_mode_requires_room_after_both_reserves(self):
        for window in (5119, 5120):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, 'reserves'):
                Config(evidence_mode='context', context_window_tokens=window).validate()
        Config(evidence_mode='context', context_window_tokens=5121).validate()
        # An inactive token mode does not invalidate existing fixed-count use.
        Config(evidence_mode='fixed', context_window_tokens=1).validate()


class ContextBudgetWebTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(embedding='demo')
        store = Mock()
        store.stats.return_value = {'units': 0, 'chunks': 0}
        store.source_catalog.return_value = []
        self.plan_mock = self.enterContext(patch('mori.web.plan', return_value={'fixture': True}))
        self.query_mock = self.enterContext(patch('mori.web.query'))
        self.embedding_mock = self.enterContext(patch('mori.web.Embeddings'))
        self.server = create_server(store, self.config, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.origin = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        _, html = self.request('/')
        self.token = re.search(r'name="mori-token" content="([a-f0-9]+)"', html.decode())[1]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, body=None):
        headers = {'Origin': self.origin, 'X-Mori-Token': getattr(self, 'token', ''),
                   'Content-Type': 'application/json'}
        request = urllib.request.Request(self.origin + path, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        try:
            response = self.opener.open(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, response.read()

    def test_state_exposes_budget_controls_and_readonly_counting_metadata(self):
        code, body = self.request('/api/state')
        self.assertEqual(code, 200)
        state = json.loads(body)
        self.assertEqual(state['settings']['evidence_mode'], 'fixed')
        self.assertEqual(state['settings']['context_window_tokens'], 100000)
        self.assertEqual(state['settings']['llm_max_output_tokens'], 4096)
        self.assertNotIn('context_tokenizer', state['settings'])
        self.assertNotIn('context_overhead_tokens', state['settings'])
        self.assertEqual(state['context_budget'], {'counting_method': 'utf8_bytes',
                                                  'counting_note': COUNTING_NOTE,
                                                  'overhead_reserve_tokens': 1024})

    def test_valid_overrides_reach_preflight_without_mutating_server_defaults(self):
        settings = {'evidence_mode': 'context', 'context_window_tokens': 150000,
                    'llm_max_output_tokens': 8000, 'jev_k': 10, 'final_passages': 20}
        self.assertEqual(self.request('/api/plan', {'settings': settings})[0], 200)
        active = self.plan_mock.call_args.args[1]
        for name, value in settings.items():
            self.assertEqual(getattr(active, name), value)
        self.assertEqual(self.config.evidence_mode, 'fixed')
        self.assertEqual(self.config.context_window_tokens, 100000)
        self.assertEqual(self.config.llm_max_output_tokens, 4096)
        self.assertEqual(self.request('/api/plan', {'settings': {}})[0], 200)
        self.assertEqual(self.plan_mock.call_args.args[1].evidence_mode, 'fixed')
        self.embedding_mock.assert_not_called()

    def test_invalid_and_server_only_settings_fail_before_provider_work(self):
        settings = [{'evidence_mode': 'unknown'}, {'context_window_tokens': True},
                    {'llm_max_output_tokens': False}, {'context_overhead_tokens': 0},
                    {'context_tokenizer': 'utf8'}, {'context_budget': {'counting_method': 'other'}},
                    {'evidence_mode': 'context', 'context_window_tokens': 5120}]
        for values in settings:
            for path in ('/api/plan', '/api/query'):
                with self.subTest(settings=values, path=path):
                    self.assertEqual(self.request(path, {'settings': values, 'question': 'Synthetic'})[0], 400)
        self.embedding_mock.assert_not_called()
        self.query_mock.assert_not_called()
        self.plan_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
