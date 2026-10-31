"""Readable parent passages and exact, token-safe multi-vector retrieval storage."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import re
import sqlite3
import tempfile
import types
import unittest

from mori.chunking import boundaries, sentence_chunks, token_spans
from mori.config import Config
from mori.ingest import chunks, ingest
from mori.providers import Embeddings, expected_embedding_fingerprint
from mori.store import Store, compatible_fingerprint


class ChildEmbeddings:
    fingerprint = {'provider': 'test', 'model': 'exact-child-test-v1'}

    def __init__(self):
        self.calls = []

    def embedding_spans(self, text):
        return token_spans(text, lambda value: len(value.split()) + 2, 30)

    def encode(self, texts):
        self.calls.extend(texts)
        return [[0.0, 1.0] if 'needle' in text else [1.0, 0.0] for text in texts]


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / 'corpus')
        self.config = Config(embedding='demo')

    def unit(self, text=None, summary=None):
        return {'id': 'unit-1', 'source_id': 'work', 'title': 'Exact source',
                'text': text or ('Dr. Smith explains this complete philosophical argument carefully. '
                                 'The second sentence preserves its qualifications and broader context.\n\n') * 80,
                'summary': summary or ('An opening discussion presents context. ' * 40) + 'The final qualification contains needle.'}

    def assert_coverage(self, text, rows):
        covered = 0
        for row in rows:
            self.assertLessEqual(row['start'], covered)
            self.assertGreater(row['end'], covered)
            self.assertEqual(row['text'], text[row['start']:row['end']])
            covered = row['end']
        self.assertEqual(covered, len(text))

    def test_defaults_are_readable_parents_and_explicit_legacy_strategy(self):
        self.assertEqual((self.config.chunk_strategy, self.config.chunk_words, self.config.chunk_overlap),
                         ('sentence', 400, 50))
        self.assertEqual(self.config.final_passages, 20)
        legacy = Config(embedding='demo', chunk_strategy='words', chunk_words=40, chunk_overlap=8)
        self.assertNotIn('retrieval_pooling', Embeddings(legacy).fingerprint)
        current = Embeddings(self.config).fingerprint
        self.assertEqual(current['retrieval_pooling'], 'max-child-cosine-v1')
        self.assertEqual(current['summary_pooling'], 'max-child-cosine-v1')
        self.assertEqual(current['embedding_span_tokens'], 64)
        self.assertEqual([r['text'] for r in chunks('one two three four', 3, 1)],
                         ['one two three', 'three four'])

    def test_parent_passages_preserve_complete_sentences_and_source_coverage(self):
        source = self.unit()['text']
        rows = sentence_chunks(source, 400, 50)
        self.assert_coverage(source, rows)
        self.assertGreater(len(rows), 1)
        for row in rows:
            self.assertGreaterEqual(len(row['text'].split()), 200 if row['boundary'] == 'end' else 300)
            self.assertLessEqual(len(row['text'].split()), 500)
            self.assertIn(row['boundary'], ('sentence', 'paragraph', 'end'))
            self.assertTrue(row['text'].startswith(('Dr. Smith', 'The second')))
            self.assertTrue(row['text'].rstrip().endswith('.'))

    def test_unicode_quotes_abbreviations_and_timestamp_preserved(self):
        source = ('[00:10:02] Dr. Émile says, “This isn’t a disconnected fragment.”\n'
                  'Prof. Smith refers to Fig. 2 and e.g. a café discussion.\n\n') * 40
        points = boundaries(source)
        self.assertNotIn(source.index('Dr.') + len('Dr. '), points)
        self.assertNotIn(source.index('Prof.') + len('Prof. '), points)
        rows = sentence_chunks(source, 80, 10)
        self.assert_coverage(source, rows)
        self.assertTrue(all(row['boundary'] != 'word-fallback' for row in rows))
        self.assertIn(5, boundaries('这是句子。这是另一句。'))

    def test_long_unpunctuated_sentence_reports_word_fallback_without_gaps(self):
        source = '  ' + ' '.join('word' + str(i) for i in range(1300)) + '\n'
        rows = sentence_chunks(source, 400, 50)
        self.assert_coverage(source, rows)
        self.assertEqual(rows[0]['boundary'], 'word-fallback')
        self.assertEqual(len(rows[0]['text'].split()), 400)
        self.assertTrue(all(len(row['text'].split()) <= 500 for row in rows))

    def test_short_final_tail_is_absorbed(self):
        source = ('This sentence consists of exactly eight words today. ' * 50)
        rows = sentence_chunks(source, 400, 50)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['text'], source)

    def test_token_children_include_prefix_and_special_tokens(self):
        source = ('Émile discusses meaning. A longer qualification follows here.\n\n' * 30)
        prefix = 'document context '
        count = lambda text: len(re.findall(r'\w+|[^\w\s]', prefix + text)) + 2
        rows = token_spans(source, count, 24)
        self.assert_coverage(source, rows)
        self.assertGreater(len(rows), 10)
        self.assertTrue(all(count(row['text']) <= 24 for row in rows))
        self.assertTrue(all(row['tokens'] == count(row['text']) for row in rows))

    def test_unbreakable_text_and_impossible_prefix_fail(self):
        with self.assertRaisesRegex(ValueError, 'unbreakable'):
            token_spans('a' * 80, lambda text: len(text) + 2, 30)
        with self.assertRaisesRegex(ValueError, 'prefix'):
            token_spans('words', lambda text: len(text) + 30, 30)

    def test_onnx_span_planning_disables_truncation_and_measures_actual_inputs(self):
        state = types.SimpleNamespace(disabled=False, padding=False)
        class Tokenizer:
            def no_truncation(self):
                state.disabled = True
            def no_padding(self):
                state.padding = False
            def encode_batch(self, texts):
                result = []
                for text in texts:
                    # Non-ASCII words and punctuation deliberately tokenize
                    # differently from whitespace word counts.
                    count = len(re.findall(r'\w+|[^\w\s]', text)) + text.count('É') + 2
                    if not state.disabled:
                        count = min(count, 2)
                    result.append(types.SimpleNamespace(attention_mask=[1] * count))
                return result
        config = Config(embedding='onnx', embedding_model=self.temp.name,
                        document_prefix='document: ', embedding_token_limit=18, embedding_span_tokens=30)
        embeddings = Embeddings(config)
        embeddings._tokenizer = Tokenizer()
        source = 'Émile explains context. A second sentence preserves qualifications. ' * 20
        rows = embeddings.embedding_spans(source)
        self.assertTrue(state.disabled)
        self.assert_coverage(source, rows)
        self.assertTrue(all(row['tokens'] <= 18 for row in rows))
        self.assertIsNone(embeddings.model)  # Tokenization does not run inference.

    def test_http_fallback_is_explicit_bytes_heuristic_and_respects_prefix(self):
        config = Config(embedding='http', document_prefix='document: ', embedding_token_limit=64)
        embeddings = Embeddings(config)
        source = 'An exact café passage has a complete thought. ' * 10
        rows = embeddings.embedding_spans(source)
        self.assert_coverage(source, rows)
        self.assertEqual(embeddings.fingerprint['embedding_spans'], 'utf8-byte-bound-no-tokenizer-v1')
        self.assertTrue(all(len((config.document_prefix + row['text']).encode()) + 2 <= 64 for row in rows))

    def test_child_vectors_persist_with_absolute_parent_and_relative_summary_offsets(self):
        provider = ChildEmbeddings()
        original = self.unit()
        result = ingest(self.store, [original], self.config, provider)
        self.assertEqual(result['added_units'], 1)
        rows = Store(self.store.root).chunks()
        self.assert_coverage(original['text'], rows)
        self.assertGreater(len(rows[0]['text'].split()), 300)
        for row in rows:
            self.assertGreater(len(row['search_vectors']), 1)
            self.assertEqual(row['vector'], row['search_vectors'][0])
            self.assertEqual(row['embedding_spans'][0]['start'], row['start'])
            self.assertEqual(row['embedding_spans'][-1]['end'], row['end'])
            self.assertEqual(len(row['search_vectors']), len(row['embedding_spans']))
        unit = self.store.units()[0]
        self.assertEqual(unit['text'], original['text'])
        self.assertEqual(unit['summary'], original['summary'])
        self.assertGreater(len(unit['summary_search_vectors']), 1)
        self.assertIn([0.0, 1.0], unit['summary_search_vectors'])
        self.assertEqual(unit['summary_embedding_spans'][-1]['end'], len(original['summary']))
        self.assertEqual(self.store.fingerprint()['summary_pooling'], 'max-child-cosine-v1')
        self.assertEqual(self.store.select_sources(['work']).chunks(), rows)
        self.assertEqual(self.store.select_sources(['work']).units(), [unit])

    def test_invalid_later_span_fails_before_any_encode_or_write(self):
        class Invalid(ChildEmbeddings):
            def embedding_spans(self, text):
                if 'bad-span' in text:
                    return [{'start': 1, 'end': len(text), 'text': text[1:]}]
                return super().embedding_spans(text)
        provider = Invalid()
        with self.assertRaisesRegex(ValueError, 'complete text'):
            ingest(self.store, [self.unit(summary='bad-span missing beginning')], self.config, provider)
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.store.stats()['units'], 0)
        self.assertIsNone(self.store.fingerprint())

    def test_dimension_failure_is_atomic(self):
        class Invalid(ChildEmbeddings):
            def encode(self, texts):
                return [[1.0, 0.0, 0.0] if 'needle' in text else [1.0, 0.0] for text in texts]
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            ingest(self.store, [self.unit()], self.config, Invalid())
        self.assertEqual(self.store.stats()['units'], 0)
        self.assertIsNone(self.store.fingerprint())

    def test_custom_adapter_without_span_method_is_explicit_whole_parent(self):
        class Legacy:
            fingerprint = {'model': 'legacy-custom'}
            def encode(self, texts):
                return [[1.0, 0.0] for text in texts]
        ingest(self.store, [self.unit()], self.config, Legacy())
        self.assertEqual(self.store.fingerprint()['embedding_spans'], 'adapter-whole-parent-v1')
        self.assertTrue(all(len(row['search_vectors']) == 1 for row in self.store.chunks()))

    def test_new_policy_missing_children_is_visible_as_empty_not_legacy_fallback(self):
        ingest(self.store, [self.unit()], self.config, ChildEmbeddings())
        with self.store.connect() as db:
            db.execute('DELETE FROM chunk_embeddings')
            db.execute('DELETE FROM summary_embeddings')
        self.assertEqual(self.store.chunks()[0]['search_vectors'], [])
        self.assertEqual(self.store.units()[0]['summary_search_vectors'], [])

    def test_legacy_database_without_child_tables_reads_without_mutation(self):
        config = replace(self.config, chunk_strategy='words', chunk_words=40, chunk_overlap=8)
        ingest(self.store, [self.unit()], config, Embeddings(config))
        with self.store.connect() as db:
            for table in ('chunk_embeddings', 'summary_embeddings', 'chunk_details'):
                db.execute('DROP TABLE ' + table)
        before = self.store.db.read_bytes()
        class ReadOnly(Store):
            @contextmanager
            def connect(self):
                db = sqlite3.connect(self.db.as_uri() + '?mode=ro', uri=True)
                db.row_factory = sqlite3.Row
                try:
                    yield db
                finally:
                    db.close()
        readonly = object.__new__(ReadOnly)
        readonly.root, readonly.db = self.store.root, self.store.db
        self.assertNotIn('search_vectors', readonly.chunks()[0])
        self.assertNotIn('summary_search_vectors', readonly.units()[0])
        self.assertEqual(readonly.db.read_bytes(), before)

    def test_policy_change_requires_new_index_before_encode(self):
        legacy = replace(self.config, chunk_strategy='words', chunk_words=40, chunk_overlap=8)
        ingest(self.store, [self.unit()], legacy, Embeddings(legacy))
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            ingest(self.store, [self.unit()], self.config, Embeddings(self.config))

    def test_new_index_cannot_be_used_with_legacy_policy_in_reverse(self):
        ingest(self.store, [self.unit()], self.config, Embeddings(self.config))
        legacy = replace(self.config, chunk_strategy='words')
        provider = Embeddings(legacy)
        provider.encode = lambda texts: self.fail('Fingerprint must reject before encoding')
        self.assertFalse(compatible_fingerprint(self.store.fingerprint(), provider.fingerprint))
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch before import'):
            ingest(self.store, [{**self.unit(), 'id': 'unit-2'}], legacy, provider)

    def test_custom_adapter_expected_fingerprint_matches_augmentation(self):
        provider = ChildEmbeddings()
        ingest(self.store, [self.unit()], self.config, provider)
        self.assertTrue(compatible_fingerprint(self.store.fingerprint(),
                                              expected_embedding_fingerprint(self.config, provider)))

    def test_parent_and_summary_span_validation_before_store_mutation(self):
        ingest(self.store, [self.unit()], self.config, ChildEmbeddings())
        unit, rows = self.store.units()[0], self.store.chunks()
        other = Store(Path(self.temp.name) / 'invalid-store')
        rows[0]['embedding_spans'][0]['start'] += 1
        with self.assertRaisesRegex(ValueError, 'complete parent'):
            other.add([unit], {unit['id']: rows}, self.store.fingerprint())
        self.assertEqual(other.stats()['units'], 0)
        self.assertIsNone(other.fingerprint())

    def test_nonunit_parent_and_summary_child_vectors_fail_before_storage(self):
        ingest(self.store, [self.unit()], self.config, ChildEmbeddings())
        for target in ('passage', 'summary'):
            with self.subTest(target=target):
                unit, rows = self.store.units()[0], self.store.chunks()
                vectors = rows[0]['search_vectors'] if target == 'passage' else unit['summary_search_vectors']
                vectors[0] = [2.0, 0.0]
                other = Store(Path(self.temp.name) / ('nonunit-' + target))
                with self.assertRaisesRegex(ValueError, 'must be normalized'):
                    other.add([unit], {unit['id']: rows}, self.store.fingerprint())
                self.assertEqual(other.stats()['units'], 0)
                self.assertIsNone(other.fingerprint())


if __name__ == '__main__':
    unittest.main()
