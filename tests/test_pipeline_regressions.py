import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mori.config import Config
from mori.ingest import ingest
from mori.pipeline import normalize_citations, plan, query, validate_answer
from mori.providers import Embeddings, LLM
from mori.store import Store


ROOT = Path(__file__).resolve().parents[1]


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def score(self, question, evidence, purpose):
        self.calls.append((purpose, json.loads(evidence)))
        return {'score': 80.0}


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='test-fixture-v1',
                             retrieval_k=10, jev_k=10, final_passages=10, rerank_k=10,
                             full_units=0, summaries=0, chunk_words=40, chunk_overlap=0,
                             min_relevance=0, summary_min_relevance=0).validate()
        self.embeddings = Embeddings(self.config)

    def add_units(self, units):
        ingest(self.store, units, self.config, self.embeddings)

    def unit(self, uid, **overrides):
        return {'id': uid, 'source_id': 'source-' + uid, 'title': 'Title ' + uid,
                'text': 'Wetland evidence ' + uid, 'summary': 'Complete summary ' + uid,
                **overrides}

    def run_query(self, **kwargs):
        self.scorer = RecordingScorer()
        result = query(self.store, 'wetland evidence', self.config, embeddings=self.embeddings,
                       scorer=self.scorer, **kwargs)
        self.run = Path(result['directory'])
        return json.loads((self.run / 'context.json').read_text(encoding='utf-8'))

    def test_disabled_summary_branch_uses_only_passage_budget_and_calls(self):
        self.add_units([self.unit('a')])
        self.config.max_rating_calls = 1
        estimate = plan(self.store, self.config)
        self.assertEqual(estimate['summary_ratings'], 0)
        self.assertEqual(estimate['maximum_rating_decisions'], 1)
        packet = self.run_query()
        self.assertEqual([kind for kind, data in self.scorer.calls], ['target passage'])
        self.assertEqual(packet['counts']['rated_summaries'], 0)
        self.assertEqual(json.loads((self.run / 'summary_ratings.json').read_text()), [])

    def test_rejected_answer_is_preserved_without_becoming_a_completed_answer(self):
        self.add_units([self.unit('a')])
        rejected = 'This generated response has an unknown citation [P999].'
        class Writer:
            def complete(self, task, prompt):
                return rejected
        with self.assertRaisesRegex(ValueError, 'unknown evidence citations'):
            query(self.store, 'wetland evidence', self.config, embeddings=self.embeddings,
                  scorer=RecordingScorer(), llm=Writer(), answer=True)
        run, = (Path(self.temp.name) / 'runs').iterdir()
        self.assertEqual((run / 'answer.rejected.md').read_text(encoding='utf-8'), rejected)
        self.assertFalse((run / 'answer.md').exists())
        self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'failed')

    def test_grouped_answer_is_saved_with_clickable_ids_and_original_output(self):
        self.add_units([self.unit('a')])
        original = 'This supported answer uses a supplied passage. [P1, physical pp. 8–11]'
        class Writer:
            def complete(self, task, prompt):
                return original
        self.run_query(llm=Writer(), answer=True)
        self.assertEqual((self.run / 'answer.raw.md').read_text(encoding='utf-8'), original)
        self.assertEqual((self.run / 'answer.md').read_text(encoding='utf-8'),
                         'This supported answer uses a supplied passage. [P1] (physical pp. 8–11)')
        self.assertFalse((self.run / 'answer.rejected.md').exists())
        self.assertEqual(json.loads((self.run / 'status.json').read_text())['status'], 'complete')

    def test_unknown_id_hidden_in_group_preserves_rejected_original(self):
        self.add_units([self.unit('a')])
        original = 'This generated answer contains an unknown grouped citation. [P1; P999]'
        class Writer:
            def complete(self, task, prompt):
                return original
        with self.assertRaisesRegex(ValueError, 'unknown evidence citations'):
            query(self.store, 'wetland evidence', self.config, embeddings=self.embeddings,
                  scorer=RecordingScorer(), llm=Writer(), answer=True)
        run, = (Path(self.temp.name) / 'runs').iterdir()
        self.assertEqual((run / 'answer.rejected.md').read_text(encoding='utf-8'), original)
        self.assertFalse((run / 'answer.md').exists())
        self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'failed')

    def test_supported_citation_forms_produce_only_individual_supplied_ids(self):
        packet = {'full_units': [{'citation': 'U1'}], 'summaries': [{'citation': f'M{i}'} for i in range(1, 6)],
                  'passages': [{'citation': 'P1'}, {'citation': 'P14'}]}
        cases = [('[U1] [P1]', '[U1] [P1]'),
                 ('[U1, physical pp. 8–11; P1]', '[U1] (physical pp. 8–11) [P1]'),
                 ('[U1, physical pp. 3–6, 11–15]', '[U1] (physical pp. 3–6, 11–15)'),
                 ('[U1, physical pp. 2, 15–16 and supplied extraction metadata; P1; M1–M5]',
                  '[U1] (physical pp. 2, 15–16 and supplied extraction metadata) [P1] [M1] [M2] [M3] [M4] [M5]'),
                 ('[M4; P14]', '[M4] [P14]'),
                 ('[M1–M5]', '[M1] [M2] [M3] [M4] [M5]'),
                 ('[M1-M3]', '[M1] [M2] [M3]'),
                 ('[M1–3]', '[M1] [M2] [M3]'),
                 ('[ordinary editorial note] [U1]', '[ordinary editorial note] [U1]')]
        for original, expected in cases:
            with self.subTest(original=original):
                result = normalize_citations('Supported claim. ' + original, packet)
                self.assertEqual(result, 'Supported claim. ' + expected)
                self.assertTrue(validate_answer(result, packet))
                self.assertEqual(normalize_citations(result, packet), result)

    def test_citation_groups_reject_unknown_ranges_and_unparsed_content(self):
        packet = {'full_units': [{'citation': 'U1'}], 'summaries': [{'citation': 'M1'}, {'citation': 'M3'}],
                  'passages': [{'citation': 'P1'}]}
        for original in ('[U1; P999]', '[M1–M3]', '[M3–M1]', '[U1–P1]', '[M1–M999999999]',
                         '[U1, unexplained locator]', '[U1 [P1]]', '[U1;]', '[U1, physical pp. 8–11; P999]',
                         '[U1, physical pp. 2, 15–16 and P999]', '[U1, physical pp. 2, 15–16; P999]'):
            with self.subTest(original=original):
                with self.assertRaisesRegex(ValueError, 'unknown evidence citations'):
                    normalize_citations(original, packet)

    def test_either_summary_output_still_enables_summary_ratings(self):
        self.add_units([self.unit('a'), self.unit('b')])
        for full_units, summaries in ((1, 0), (0, 1)):
            with self.subTest(full_units=full_units, summaries=summaries):
                self.config.full_units, self.config.summaries = full_units, summaries
                self.assertEqual(plan(self.store, self.config)['summary_ratings'], 2)
                self.assertEqual(self.run_query()['counts']['rated_summaries'], 2)

    def test_same_words_keep_distinct_unit_context_and_survive_full_unit_exclusion(self):
        self.add_units([self.unit('a', title='Supporting argument', text='The exact same quoted evidence.'),
                        self.unit('b', title='A critical response', text='The exact same quoted evidence.')])
        self.config.full_units = 1
        packet = self.run_query()
        self.assertEqual(packet['counts']['retrieved'], 2)
        self.assertEqual(packet['full_units'][0]['id'], 'a')
        self.assertEqual([row['unit']['id'] for row in packet['passages']], ['b'])
        titles = {data['title'] for kind, data in self.scorer.calls if kind == 'target passage'}
        self.assertEqual(titles, {'Supporting argument', 'A critical response'})

    def test_repeated_text_in_one_unit_is_still_deduplicated(self):
        self.config.chunk_words = 4
        self.add_units([self.unit('a', text='The same exact words. The same exact words.')])
        self.assertEqual(self.run_query()['counts']['retrieved'], 1)

    def test_provenance_reaches_full_summary_and_passage_handoffs(self):
        metadata = {uid: {'author': 'Author ' + uid, 'url': 'https://example.invalid/' + uid,
                         'extraction_limitations': 'Caveat ' + uid,
                         'summary_provenance': {'model': 'summary-model-' + uid},
                         'note': 'Quotes "quoted"\nand caf\u00e9', 'page_range': [1, 5]}
                    for uid in ('a', 'b', 'c')}
        self.add_units([self.unit(uid, metadata=metadata[uid]) for uid in metadata])
        self.config.full_units, self.config.summaries = 1, 1
        packet = self.run_query()
        self.assertTrue(all(packet[key] for key in ('full_units', 'summaries', 'passages')))
        for filename in ('context.md', 'answer_prompt.txt'):
            rendered = (self.run / filename).read_text(encoding='utf-8')
            exported = [json.loads(line.split(': ', 1)[1]) for line in rendered.splitlines()
                        if line.startswith('Source metadata (untrusted): ')]
            self.assertEqual(len(exported), 4)  # one full unit, one summary, two passages
            for record in exported:
                self.assertEqual(record, metadata[record['author'].rsplit(' ', 1)[1]])
            for uid in ('a', 'b', 'c'):
                for value in ('Author ' + uid, 'https://example.invalid/' + uid,
                              'Caveat ' + uid, 'summary-model-' + uid):
                    self.assertIn(value, rendered)

    def test_metadata_counts_toward_context_budget(self):
        self.add_units([self.unit('a', metadata={'extraction_limitations': 'Caveat ' * 200})])
        self.config.context_max_chars = 600
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            self.run_query()
        run = next((self.store.root / 'runs').iterdir())
        self.assertFalse((run / 'context.json').exists())

    def test_clearing_requested_dimensions_fails_before_query_embedding(self):
        self.config.embedding_dimensions = 256
        self.add_units([self.unit('a')])
        self.config.embedding_dimensions = 0
        with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected provider call')):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                self.run_query()

    def test_legacy_unspecified_dimensions_remain_compatible(self):
        self.add_units([self.unit('a')])
        fingerprint = self.store.fingerprint()
        fingerprint.pop('requested_dimensions', None)
        with self.store.connect() as db:
            db.execute("UPDATE meta SET value=? WHERE key='embedding'", (json.dumps(fingerprint),))
        self.assertEqual(self.run_query()['counts']['passages'], 1)

    def test_config_rejects_nonstring_provider_fields(self):
        for name, value in (('embedding_model', 12), ('query_prefix', []), ('llm_model', None),
                            ('jev_key_env', ['KEY']), ('embedding_url', False)):
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, name + ' must be a string'):
                    dataclasses.replace(self.config, **{name: value}).validate()

    def test_only_active_final_passage_cap_constrains_shortlist(self):
        config = Config(retrieval_k=10, jev_k=10, final_passages=10).validate()
        self.assertEqual(config.rerank_k, 20)
        with self.assertRaisesRegex(ValueError, 'active final passage cap'):
            dataclasses.replace(config, rerank_enabled=True).validate()
        config = Config(retrieval_k=10, jev_k=10, rerank_enabled=True, rerank_k=5).validate()
        self.assertEqual(config.final_passages, 100)
        with self.assertRaisesRegex(ValueError, 'active final passage cap'):
            dataclasses.replace(config, rerank_enabled=False).validate()

    def test_missing_answer_or_rerank_provider_fails_before_any_provider_calls(self):
        self.add_units([self.unit('a')])
        for answer, rerank in ((True, False), (False, True)):
            for llm in (None, LLM(self.config)):
                with self.subTest(answer=answer, rerank=rerank, explicit_llm=llm is not None):
                    self.config.rerank_enabled = rerank
                    with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected embedding call')):
                        with self.assertRaisesRegex(ValueError, 'Configure an LLM provider'):
                            self.run_query(llm=llm, answer=answer)
                    self.assertEqual(self.scorer.calls, [])
                    self.assertFalse((self.store.root / 'runs').exists())

    def test_injected_writer_is_allowed_with_unconfigured_default_provider(self):
        self.add_units([self.unit('a')])

        class Writer:
            def complete(self, task, prompt):
                return 'This supported answer uses the supplied passage. [P1]'

        self.assertEqual(self.run_query(llm=Writer(), answer=True)['counts']['passages'], 1)


if __name__ == '__main__':
    unittest.main()
