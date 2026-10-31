"""Behavioral expansion boundaries with synthetic sources and no hosted calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mori.config import Config
from mori.ingest import ingest
from mori.pipeline import plan, query
from mori.providers import Embeddings
from mori.store import Store


QUESTION = 'How does the original argument work?'
VARIANTS = {key: 'EXPANSION_SENTINEL_' + key for key in
            ('improved', 'specific', 'broad', 'hypothetical_answer')}
ROOT = Path(__file__).resolve().parents[1]


def axis(index):
    return [float(i == index) for i in range(256)]


class QueryExpansionTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='expansion-fixture-v1', llm='http',
                             retrieval_k=5, jev_k=5, final_passages=5, rerank_k=5,
                             full_units=0, summaries=0, workers=1, min_relevance=0,
                             summary_min_relevance=0, chunk_strategy='words',
                             chunk_words=4, chunk_overlap=0).validate()
        self.embeddings = Embeddings(self.config)
        units = [{'id': uid, 'source_id': 'selected' if uid != 'z' else 'excluded',
                  'title': 'Unit ' + uid,
                  'text': ('Identical evidence across units.' if uid in ('b', 'c')
                           else 'Original evidence for ' + uid + '.'),
                  'summary': 'Complete synthetic summary ' + uid}
                 for uid in ('a', 'b', 'c', 'd', 'e', 'f', 'z')]
        # A repeated passage in a single unit must consume just one rating.
        units[0]['text'] = 'Repeated original evidence words. Repeated original evidence words.'
        ingest(self.store, units, self.config, self.embeddings)
        with self.store.connect() as db:
            for uid, index in (('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4), ('f', 0), ('z', 1)):
                values = axis(index)
                if uid == 'f':
                    values = [0.8 * x + 0.6 * y for x, y in zip(axis(0), axis(5))]
                if uid == 'z':
                    values = [0.8 * x + 0.6 * y for x, y in zip(axis(1), axis(6))]
                db.execute('UPDATE chunks SET vector=? WHERE unit_id=?', (json.dumps(values), uid))
                db.execute('UPDATE units SET summary_vector=? WHERE id=?', (json.dumps(values), uid))
        self.encoded = []

        def encode(texts, query=False):
            self.assertTrue(query)
            self.encoded.extend(texts)
            return [axis(([QUESTION] + list(VARIANTS.values())).index(text)) for text in texts]

        self.enterContext(patch.object(self.embeddings, 'encode', side_effect=encode))
        self.scorer = Mock()
        self.scorer.score.return_value = {'score': 80, 'cached': False}
        self.llm = Mock()
        self.llm.json.return_value = VARIANTS.copy()
        self.llm.complete.return_value = 'The supplied evidence supports this answer. [P1]'

    def run_query(self, store=None, answer=False):
        result = query(store or self.store, QUESTION, self.config,
                       embeddings=self.embeddings, scorer=self.scorer, llm=self.llm, answer=answer)
        self.run = Path(result['directory'])
        return json.loads((self.run / 'context.json').read_text(encoding='utf-8'))

    def retrieved_units(self):
        return [row['unit_id'] for row in json.loads((self.run / 'retrieved.json').read_text())]

    def test_disabled_is_default_and_does_not_call_expansion_model(self):
        self.assertFalse(self.config.query_expansion)
        self.run_query()
        self.assertEqual(self.encoded, [QUESTION])
        self.llm.json.assert_not_called()
        self.assertEqual(self.retrieved_units()[:2], ['a', 'f'])

    def test_interleaved_union_is_bounded_distinct_and_keeps_cross_unit_attribution(self):
        self.config.query_expansion = True
        packet = self.run_query()
        self.assertEqual(self.encoded, [QUESTION] + list(VARIANTS.values()))
        self.assertEqual(self.retrieved_units(), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(packet['counts']['retrieved'], 5)
        self.assertEqual(self.scorer.score.call_count, 5)
        self.assertEqual({row['unit']['id'] for row in packet['passages']}, {'a', 'b', 'c', 'd', 'e'})
        self.assertEqual(self.llm.json.call_count, 1)
        self.assertEqual(self.llm.json.call_args.args[0], 'query_expansion')
        rated = json.loads((self.run / 'passage_ratings.json').read_text())
        self.assertEqual(rated[0]['unit_id'], 'a')
        self.assertEqual([row['id'] for row in rated[1:]], sorted(row['id'] for row in rated[1:]))
        self.assertEqual([row['similarity'] for row in rated], [1.0, 0.0, 0.0, 0.0, 0.0])

    def test_summary_preselection_interleaves_and_original_question_reaches_all_decision_stages(self):
        self.config.query_expansion = True
        self.config.summaries = 3
        self.config.summary_candidates = 3
        self.config.rerank_enabled = True

        def respond(task, prompt):
            if task == 'query_expansion':
                return VARIANTS.copy()
            self.assertEqual(task, 'rerank')
            data = json.loads(prompt[prompt.index('\n') + 1:])
            self.assertEqual(data['question'], QUESTION)
            self.assertNotIn('EXPANSION_SENTINEL', prompt)
            return {'ids': [row['id'] for row in data['passages']]}

        self.llm.json.side_effect = respond
        packet = self.run_query(self.store.select_sources(['selected']), answer=True)
        rated = json.loads((self.run / 'summary_ratings.json').read_text())
        self.assertEqual({row['id'] for row in rated}, {'a', 'b', 'c'})
        self.assertEqual(self.retrieved_units(), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual({call.args[2] for call in self.scorer.score.call_args_list},
                         {'target passage', 'complete source summary'})
        for call in self.scorer.score.call_args_list:
            self.assertEqual(call.args[0], QUESTION)
            self.assertNotIn('EXPANSION_SENTINEL', call.args[1])
        self.assertEqual(packet['question'], QUESTION)
        self.assertNotIn('EXPANSION_SENTINEL', json.dumps(packet))
        for name in ('answer_prompt.txt', 'context.md', 'answer.md'):
            self.assertNotIn('EXPANSION_SENTINEL', (self.run / name).read_text(encoding='utf-8'))
        self.assertNotIn('EXPANSION_SENTINEL', self.llm.complete.call_args.args[1])
        self.assertEqual([call.args[0] for call in self.llm.json.call_args_list], ['query_expansion', 'rerank'])
        self.assertEqual({row['unit']['source_id'] for row in packet['passages']}, {'selected'})

    def test_failed_or_malformed_expansion_falls_back_to_original_retrieval(self):
        self.config.query_expansion = True
        for malformed in (RuntimeError('synthetic provider failure'), None, [],
                          {'improved': 42}):
            with self.subTest(response=malformed):
                self.encoded.clear()
                self.llm.reset_mock()
                self.llm.json.side_effect = malformed if isinstance(malformed, Exception) else None
                self.llm.json.return_value = malformed
                self.run_query()
                self.assertEqual(self.encoded, [QUESTION])
                self.assertEqual(self.retrieved_units()[:2], ['a', 'f'])
                self.assertEqual(self.llm.json.call_count, 1)

    def test_partial_expansion_salvages_only_usable_distinct_strings(self):
        self.config.query_expansion = True
        self.llm.json.return_value = {'improved': VARIANTS['improved'], 'specific': [],
                                      'broad': ' ', 'hypothetical_answer': 'x' * 4001}
        self.run_query()
        self.assertEqual(self.encoded, [QUESTION, VARIANTS['improved']])
        self.assertEqual(self.llm.json.call_count, 1)

    def test_duplicate_variants_do_not_waste_query_embeddings(self):
        self.config.query_expansion = True
        self.llm.json.return_value = {'improved': '  ' + QUESTION + '  ',
                                      'specific': VARIANTS['specific'],
                                      'broad': '  ' + VARIANTS['specific'] + '  ',
                                      'hypothetical_answer': QUESTION}
        self.run_query()
        self.assertEqual(self.encoded, [QUESTION, VARIANTS['specific']])
        self.assertEqual(self.llm.json.call_count, 1)
        self.assertEqual(len(self.retrieved_units()), 5)
        self.assertEqual(len(set(self.retrieved_units())), 5)

    def test_unembeddable_generated_query_retries_only_the_original(self):
        self.config.query_expansion = True
        def limited(texts, query):
            if len(texts) > 1:
                raise ValueError('Synthetic input exceeds the embedding token limit')
            return [axis(0)]
        self.embeddings.encode.side_effect = limited
        packet = self.run_query()
        self.assertEqual([c.args[0] for c in self.embeddings.encode.call_args_list],
                         [[QUESTION, *VARIANTS.values()], [QUESTION]])
        self.assertEqual(self.llm.json.call_count, 1)
        self.assertTrue(all(c.args[0] == QUESTION for c in self.scorer.score.call_args_list))
        self.assertTrue(any('original question only' in s for s in packet['notices']))
        used = json.loads((self.run / 'query_expansion.json').read_bytes())
        self.assertEqual(used['variants'], [{'kind': 'original', 'text': QUESTION}])

    def test_plan_reports_expansion_ceiling_without_initializing_providers(self):
        with patch('mori.pipeline.Embeddings', side_effect=AssertionError('Provider initialized')), \
                patch('mori.pipeline.LLM', side_effect=AssertionError('Provider initialized')), \
                patch('mori.pipeline.Jev', side_effect=AssertionError('Provider initialized')):
            for enabled, embeddings, model_calls in ((False, 1, 0), (True, 5, 1)):
                self.config.query_expansion = enabled
                estimate = plan(self.store, self.config)
                self.assertEqual(estimate['maximum_query_embeddings'], embeddings)
                self.assertEqual(estimate['expansion_model_calls'], model_calls)
                self.assertEqual(estimate['candidate_passages'], 5)
        self.assertEqual(self.encoded, [])
        self.llm.json.assert_not_called()
        self.scorer.score.assert_not_called()


if __name__ == '__main__':
    unittest.main()
