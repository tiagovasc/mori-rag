import copy
import json
from pathlib import Path
import tempfile
import unittest

from mori.budget import ContextBudget
from mori.config import Config
from mori.ingest import ingest
from mori.pipeline import answer_prompt, plan, query, render_context
from mori.providers import Embeddings
from mori.store import Store


ROOT = Path(__file__).resolve().parents[1]


class Scorer:
    def __init__(self, scores=None):
        self.scores = scores or {}

    def score(self, question, evidence, purpose):
        data = json.loads(evidence)
        return {'score': self.scores.get(data['title'], 80.0)}


class Writer:
    def __init__(self):
        self.calls = []
        self.rerank_order = []

    def json(self, task, prompt):
        self.calls.append(task)
        data = json.loads(prompt[prompt.index('{"question"'):])
        self.rerank_order = [row['id'] for row in reversed(data['passages'])]
        return {'ids': self.rerank_order}

    def complete(self, task, prompt):
        self.calls.append(task)
        return 'Supported synthetic answer. [U1]'


class ContextBudgetPipelineTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='context-budget-test',
                             evidence_mode='context', full_units=0, summaries=0,
                             chunk_strategy='words', chunk_words=2000, chunk_overlap=0,
                             context_max_chars=1000000, llm_max_input_chars=1000000,
                             min_relevance=0, summary_min_relevance=0).validate()
        self.embeddings = Embeddings(self.config)
        self.scorer = Scorer()

    @staticmethod
    def unit(uid, **overrides):
        return {'id': uid, 'source_id': 'source-' + uid, 'title': uid,
                'text': 'Complete synthetic evidence for ' + uid + '.',
                'summary': 'Complete synthetic summary for ' + uid + '.', **overrides}

    def add_units(self, units):
        ingest(self.store, units, self.config, self.embeddings)

    def run_query(self, **kwargs):
        result = query(self.store, 'What does the synthetic evidence say?', self.config,
                       embeddings=self.embeddings, scorer=self.scorer, **kwargs)
        self.run = Path(result['directory'])
        return json.loads((self.run / 'context.json').read_text(encoding='utf-8'))

    def fit_exactly(self, packet):
        self.config.context_window_tokens = (ContextBudget.count(answer_prompt(packet))
                                             + self.config.llm_max_output_tokens
                                             + self.config.context_overhead_tokens)

    def test_exact_boundary_counts_the_complete_saved_prompt(self):
        self.add_units([self.unit('a')])
        initial = self.run_query()
        self.fit_exactly(initial)
        packet = self.run_query()
        budget = packet['context_budget']
        prompt = (self.run / 'answer_prompt.txt').read_bytes().decode('utf-8')
        self.assertEqual(packet['counts']['passages'], 1)
        self.assertEqual(budget['remaining_tokens'], 0)
        self.assertEqual(budget['prompt_tokens'], len(prompt.encode('utf-8')))
        self.assertEqual(prompt, answer_prompt(packet))

    def test_saved_prompt_preserves_mixed_newlines_and_matches_provider_input(self):
        original = 'First line.\r\nSecond line.\rThird line.\nUnicode café.\r\nEnd.'
        self.config.full_units, self.config.summaries = 1, 0
        self.add_units([self.unit('mixed', text=original)])
        class CaptureWriter:
            def complete(self, task, prompt):
                self.prompt = prompt
                return 'Supported synthetic answer. [U1]'
        for mode in ('context', 'fixed'):
            with self.subTest(mode=mode):
                self.config.evidence_mode = mode
                writer = CaptureWriter()
                packet = self.run_query(llm=writer, answer=True)
                raw_prompt = (self.run / 'answer_prompt.txt').read_bytes()
                self.assertEqual(raw_prompt, answer_prompt(packet).encode('utf-8'))
                self.assertEqual(raw_prompt.decode('utf-8'), writer.prompt)
                self.assertEqual((self.run / 'context.md').read_bytes(), render_context(packet).encode('utf-8'))
                self.assertEqual(packet['full_units'][0]['text'], original)
                self.assertEqual(self.store.units()[0]['text'], original)
                if mode == 'context':
                    self.assertEqual(packet['context_budget']['prompt_tokens'], len(raw_prompt))

    def test_context_mode_exceeds_fixed_caps_with_and_without_reranking(self):
        self.config.full_units, self.config.summaries = 1, 5
        self.config.final_passages, self.config.rerank_k = 2, 3
        self.add_units([self.unit(f'u{index:02}') for index in range(30)])
        for rerank in (False, True):
            with self.subTest(rerank=rerank):
                self.config.rerank_enabled = rerank
                writer = Writer()
                packet = self.run_query(llm=writer)
                self.assertEqual(packet['counts']['passages'], 29)
                self.assertEqual(packet['counts']['full_units'], 1)
                self.assertEqual(packet['counts']['summaries'], 5)
                full_ids = {row['id'] for row in packet['full_units']}
                self.assertFalse(full_ids.intersection(row['id'] for row in packet['summaries']))
                self.assertFalse(full_ids.intersection(row['unit']['id'] for row in packet['passages']))
                if rerank:
                    self.assertEqual([row['id'] for row in packet['passages']], writer.rerank_order)
                self.assertEqual(writer.calls, ['rerank'] if rerank else [])

    def test_oversized_candidate_is_skipped_and_smaller_candidates_fill_budget(self):
        self.add_units([self.unit('large', text='Long synthetic content. ' * 500),
                        self.unit('small-a'), self.unit('small-b')])
        self.scorer = Scorer({'large': 99, 'small-a': 80, 'small-b': 70})
        expected = copy.deepcopy(self.run_query())
        expected['passages'] = expected['passages'][1:]
        for index, row in enumerate(expected['passages'], 1):
            row['citation'] = f'P{index}'
        self.fit_exactly(expected)
        packet = self.run_query()
        self.assertEqual([row['unit']['id'] for row in packet['passages']], ['small-a', 'small-b'])
        self.assertEqual([row['citation'] for row in packet['passages']], ['P1', 'P2'])
        self.assertEqual(packet['context_budget']['skipped_passages'], 1)
        self.assertEqual(packet['context_budget']['remaining_tokens'], 0)
        self.assertEqual(packet['context_budget']['limiting_constraints'], ['context_window_tokens'])

    def test_mandatory_bundle_overflow_fails_before_answer_and_publication(self):
        self.config.full_units, self.config.summaries = 1, 1
        self.add_units([self.unit('a'), self.unit('b')])
        base = copy.deepcopy(self.run_query())
        base['passages'] = []
        self.fit_exactly(base)
        self.config.context_window_tokens -= 1
        self.config.rerank_enabled = True
        before = set((self.store.root / 'runs').iterdir())
        writer = Writer()
        with self.assertRaisesRegex(ValueError, 'Required complete units and summaries.*no truncation'):
            self.run_query(llm=writer, answer=True)
        self.assertEqual(writer.calls, [])
        failed, = set((self.store.root / 'runs').iterdir()) - before
        self.assertFalse((failed / 'context.json').exists())
        self.assertFalse((failed / 'answer.md').exists())
        self.assertEqual(json.loads((failed / 'status.json').read_text())['status'], 'failed')

    def test_mandatory_bundle_can_use_entire_allowance_without_passages(self):
        self.config.full_units, self.config.summaries = 1, 1
        self.add_units([self.unit('a'), self.unit('b')])
        base = copy.deepcopy(self.run_query())
        base['passages'] = []
        self.fit_exactly(base)
        packet = self.run_query()
        self.assertEqual(packet['counts']['passages'], 0)
        self.assertEqual(packet['context_budget']['remaining_tokens'], 0)
        self.assertEqual(packet['context_budget']['skipped_passages'], 1)
        self.assertEqual(packet['full_units'], base['full_units'])
        self.assertEqual(packet['summaries'], base['summaries'])

    def test_metadata_unicode_and_neighbor_expansion_count_without_truncation(self):
        self.config.chunk_words, self.config.answer_neighbors = 4, 1
        text = 'Alpha beta gamma delta. Café 猫 λ évidence. Iota kappa lambda mu.'
        self.add_units([self.unit('a', text=text, metadata={'author': '猫', 'note': 'é' * 100})])
        self.config.jev_k = 1
        packet = self.run_query()
        self.fit_exactly(packet)
        exact = self.run_query()
        passage, = exact['passages']
        self.assertGreater(len(passage['context']['text']), len(passage['target']['text']))
        self.assertEqual(passage['context']['text'], text[passage['context']['start']:passage['context']['end']])
        prompt = answer_prompt(exact)
        self.assertIn('é' * 100, prompt)
        self.assertEqual(exact['context_budget']['prompt_tokens'], len(prompt.encode('utf-8')))
        self.assertGreater(exact['context_budget']['prompt_tokens'], len(prompt))
        self.assertEqual(exact['context_budget']['remaining_tokens'], 0)

    def test_character_guards_pack_optional_passages_and_fail_mandatory_bundle(self):
        self.add_units([self.unit('large', text='Long synthetic content. ' * 500), self.unit('small')])
        self.scorer = Scorer({'large': 99, 'small': 80})
        expected = copy.deepcopy(self.run_query())
        expected['passages'] = expected['passages'][1:]
        expected['passages'][0]['citation'] = 'P1'
        for guard, allowance in (('context_max_chars', len(render_context(expected))),
                                 ('llm_max_input_chars', len(answer_prompt(expected)))):
            with self.subTest(guard=guard):
                self.config.context_max_chars = self.config.llm_max_input_chars = 1000000
                setattr(self.config, guard, allowance)
                packet = self.run_query()
                self.assertEqual([row['unit']['id'] for row in packet['passages']], ['small'])
                self.assertEqual(packet['context_budget']['limiting_constraints'], [guard])
                self.assertGreater(packet['context_budget']['remaining_tokens'], 0)
                self.config.full_units = 1
                with self.assertRaisesRegex(ValueError, guard):
                    self.run_query()
                self.config.full_units = 0

    def test_explicit_shortlist_and_relevance_threshold_still_apply(self):
        self.config.jev_k = 3
        self.config.min_relevance = 50
        self.add_units([self.unit(f'u{index}') for index in range(6)])
        self.scorer = Scorer({'u0': 0})
        packet = self.run_query()
        self.assertEqual(packet['counts']['passages'], 3)
        self.assertEqual(packet['context_budget']['candidate_passages'], 3)
        self.assertEqual(packet['context_budget']['shortlist_limit'], 3)
        self.assertNotIn('u0', [row['unit']['id'] for row in packet['passages']])
        self.assertEqual(packet['context_budget']['selection_stop_reason'], 'candidate_pool_exhausted')

    def test_fixed_mode_counts_and_failure_behavior_are_preserved(self):
        self.config.evidence_mode = 'fixed'
        self.config.final_passages = 2
        self.add_units([self.unit(f'u{index}') for index in range(5)])
        packet = self.run_query()
        self.assertEqual(packet['counts']['passages'], 2)
        self.assertNotIn('context_budget', packet)
        self.config.context_max_chars = len(render_context(packet)) - 1
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            self.run_query()

    def test_no_fitting_evidence_reports_budget_failure_without_partial_packet(self):
        self.add_units([self.unit('a')])
        base = copy.deepcopy(self.run_query())
        base['passages'] = []
        self.fit_exactly(base)
        before = set((self.store.root / 'runs').iterdir())
        with self.assertRaisesRegex(ValueError, 'No relevant whole passages fit.*no truncation'):
            self.run_query()
        failed, = set((self.store.root / 'runs').iterdir()) - before
        self.assertFalse((failed / 'context.json').exists())

    def test_insufficient_available_summaries_are_reported_without_overlap(self):
        self.config.full_units, self.config.summaries = 1, 5
        self.add_units([self.unit('a'), self.unit('b')])
        packet = self.run_query()
        self.assertEqual(packet['counts']['summaries'], 1)
        self.assertNotEqual(packet['full_units'][0]['id'], packet['summaries'][0]['id'])
        self.assertTrue(any('Requested 5 summaries; 1 available' in text for text in packet['notices']))

    def test_plan_describes_candidate_limit_and_reserved_context(self):
        self.add_units([self.unit('a')])
        result = plan(self.store, self.config)
        self.assertEqual(result['evidence_mode'], 'context')
        self.assertEqual(result['final_passage_cap'], self.config.jev_k)
        self.assertEqual(result['final_passage_cap_kind'], 'candidate_pool')
        self.assertEqual(result['context_budget']['input_budget_tokens'], 100000 - 4096 - 1024)
        self.assertNotIn('prompt_tokens', result['context_budget'])


if __name__ == '__main__':
    unittest.main()
