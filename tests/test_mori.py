import dataclasses
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import urllib.request
import urllib.response
import zipfile
from email.message import Message
from unittest.mock import Mock, patch

from mori.config import Config, load
from mori.ingest import chunks, extract, ingest, load_file
from mori.pipeline import answer_prompt, plan, query, validate_answer
from mori.providers import Embeddings, Jev, LLM, http_json
from mori.store import Store, digest, validate_unit, vector

ROOT = Path(__file__).resolve().parents[1]


class Scorer:
    def __init__(self):
        self.calls = 0

    def score(self, question, text, purpose='passage'):
        self.calls += 1
        return {'score': 80.0, 'label': 'Very strong', 'cached': False, 'model': 'test-only'}


class Writer:
    def __init__(self):
        self.calls = []

    def json(self, task, prompt):
        self.calls.append(task)
        data = json.loads(prompt[prompt.index('{"question"'):])
        return {'ids': [r['id'] for r in reversed(data['passages'])]}

    def complete(self, task, prompt):
        self.calls.append(task)
        return 'The fictional notes treat upstream storage and downstream responsibility as linked considerations. [U1]'


class MoriTests(unittest.TestCase):
    def setUp(self):
        # Explicit workspace test root avoids Windows restricted temp-folder issues.
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.config = Config(embedding='demo', embedding_model='test-fixture-v1', chunk_words=40, chunk_overlap=8,
                             min_relevance=0, summary_min_relevance=0).validate()
        self.embeddings = Embeddings(self.config)
        self.units = load_file(ROOT / 'examples/corpus.jsonl')
        ingest(self.store, self.units, self.config, self.embeddings)

    def run_query(self, **kwargs):
        return query(self.store, 'Why restore wetlands before raising floodwalls?', self.config,
                     embeddings=self.embeddings, scorer=Scorer(), **kwargs)

    def packet(self, result):
        return json.loads((Path(result['directory']) / 'context.json').read_text(encoding='utf-8'))

    def test_default_counts_and_reranker_off(self):
        default = Config()
        self.assertEqual((default.retrieval_k, default.jev_k, default.final_passages, default.full_units, default.summaries), (500, 100, 100, 1, 5))
        writer = Writer()
        result = self.run_query(llm=writer)
        self.assertEqual(writer.calls, [])
        self.assertEqual((result['counts']['full_units'], result['counts']['summaries']), (1, 5))

    def test_offsets_and_full_sources_exact(self):
        packet = self.packet(self.run_query())
        originals = {u['id']: u for u in self.units}
        for unit in packet['full_units']:
            self.assertEqual(unit['text'], originals[unit['id']]['text'])
        for row in packet['passages']:
            target = row['target']
            self.assertEqual(target['text'], originals[row['unit']['id']]['text'][target['start']:target['end']])

    def test_full_unit_exclusion_backfill_and_unique_summaries(self):
        packet = self.packet(self.run_query())
        full = {u['id'] for u in packet['full_units']}
        summary_ids = [u['id'] for u in packet['summaries']]
        self.assertEqual(len(summary_ids), len(set(summary_ids)))
        self.assertFalse(full.intersection(summary_ids))
        self.assertFalse(full.intersection(r['unit']['id'] for r in packet['passages']))

    def test_neighbors_never_cross_units(self):
        self.config.answer_neighbors = 2
        packet = self.packet(self.run_query())
        chunks_by_id = {r['id']: r for r in self.store.chunks()}
        for row in packet['passages']:
            for neighbor in row['neighbors']:
                self.assertEqual(chunks_by_id[neighbor['id']]['unit_id'], row['unit']['id'])

    def test_context_switch_changes_jev_input(self):
        seen = []
        scorer = Scorer()
        original = scorer.score
        scorer.score = lambda q, t, purpose: (seen.append((purpose, json.loads(t))) or original(q, t, purpose))
        self.config.rating_neighbors = 1
        query(self.store, 'wetlands', self.config, embeddings=self.embeddings, scorer=scorer)
        self.assertTrue(any(data['surrounding_context'] for kind, data in seen if kind == 'target passage'))

    def test_atomic_import_failure_preserves_database(self):
        before = self.store.stats()
        duplicate = {**self.units[0], 'text': 'Changed historical source'}
        with self.assertRaises(ValueError):
            ingest(self.store, [duplicate], self.config, self.embeddings)
        self.assertEqual(self.store.stats(), before)

    def test_unchanged_import_is_idempotent_without_embeddings(self):
        with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected embedding call')):
            result = ingest(self.store, self.units, self.config, self.embeddings)
        self.assertEqual(result['added_units'], 0)

    def test_fingerprint_mismatch_blocks_query(self):
        self.config.query_prefix = 'different model semantics: '
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_query()

    def test_context_overflow_does_not_publish_partial_packet(self):
        self.config.context_max_chars = 100
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            self.run_query()
        run = next((self.store.root / 'runs').iterdir())
        self.assertFalse((run / 'context.json').exists())
        self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'failed')

    def test_optional_reranker_and_answer(self):
        self.config.rerank_enabled = True
        self.config.rerank_k = 3
        writer = Writer()
        result = self.run_query(llm=writer, answer=True)
        self.assertEqual(result['counts']['passages'], 3)
        self.assertEqual(writer.calls, ['rerank', 'answer'])

    def test_invalid_reranker_ids_fail_without_fallback(self):
        self.config.rerank_enabled = True
        writer = Writer()
        writer.json = lambda *args: {'ids': ['fabricated']}
        with self.assertRaisesRegex(ValueError, 'every supplied ID'):
            self.run_query(llm=writer)

    def test_no_irrelevant_full_unit_fallback(self):
        self.config.summary_min_relevance = 99
        with self.assertRaisesRegex(ValueError, 'Not enough relevant full units'):
            self.run_query()

    def test_decision_budget_blocks_all_provider_calls(self):
        self.config.max_rating_calls = 1
        with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected provider call')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                self.run_query()

    def test_shortfalls_are_disclosed_not_padded(self):
        packet = self.packet(self.run_query())
        self.assertTrue(packet['notices'])
        self.assertLess(packet['counts']['passages'], 100)

    def test_answer_rejects_unknown_ids_and_uncited_paragraph(self):
        packet = self.packet(self.run_query())
        with self.assertRaises(ValueError):
            validate_answer('Invented claim. [P999]', packet)
        with self.assertRaises(ValueError):
            validate_answer('One claim. [U1]\n\n' + 'Uncited assertion ' * 20, packet)

    def test_summaries_never_claim_original_was_read(self):
        prompt = answer_prompt(self.packet(self.run_query()))
        self.assertIn('secondary account', prompt)
        self.assertIn('Summary only:', prompt)

    def test_missing_summary_stops_ingestion(self):
        with self.assertRaisesRegex(ValueError, 'summary'):
            ingest(self.store, [{**self.units[0], 'id': 'new', 'summary': ''}], self.config, self.embeddings)

    def test_html_and_unicode_offsets(self):
        self.assertNotIn('malicious()', extract(b'<h1>Title</h1><script>malicious()</script><p>Body</p>', '.html'))
        text = 'Résumé αβγ\n\nThe river has water.'
        for row in chunks(text, 3, 1):
            self.assertEqual(row['text'], text[row['start']:row['end']])

    def test_jsonl_preserves_literal_unicode_line_separators(self):
        text = 'Original\u0085next line\u2028line separator\u2029paragraph separator\nordinary newline\rreturn.'
        rows = [{**self.units[0], 'id': 'unicode-source', 'text': text,
                 'summary': 'Summary\u2028with a literal separator.', 'metadata': {'note': '\u0085\u2029'}},
                {**self.units[1], 'id': 'second-source'}]
        for index, newline in enumerate(('\n', '\r\n')):
            with self.subTest(record_separator=repr(newline)):
                path = Path(self.temp.name) / f'unicode-{index}.jsonl'
                payload = newline.join(json.dumps(row, ensure_ascii=False) for row in rows)
                payload = ('\ufeff' if index else '') + payload + newline
                path.write_bytes(payload.encode('utf-8'))
                self.assertEqual(load_file(path), rows)
                self.assertEqual(path.read_bytes(), payload.encode('utf-8'))

    def test_all_config_examples_validate(self):
        for path in (ROOT / 'examples').glob('*.toml'):
            load(path)
        for values in ({'rubric': ['one'] * 20}, {'min_relevance': float('nan')}, {'rerank_enabled': 'false'}, {'workers': 0}):
            with self.assertRaises(ValueError):
                dataclasses.replace(self.config, **values).validate()

    def test_jev_fractional_score_and_cache(self):
        p = {str(i): 0.0 for i in range(10)}
        p['7'], p['8'] = 0.3, 0.7
        response = {'answers': {'relevance': {'type': 'score', 'score': 7.7, 'probabilities': p, 'confidence': 0.5}}, 'model': 'resolved-version'}
        scorer = Jev(self.config, self.store)
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'unit-test-not-a-secret'}), patch('mori.providers.http_json', return_value=response) as call:
            a = scorer.score('wetlands', 'source')
            b = scorer.score('wetlands', 'source')
        self.assertAlmostEqual(a['score'], 85.5555555555)
        self.assertEqual(a['modal_level'], 8)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(b['cached'])

    def test_bad_jev_probabilities_not_cached(self):
        response = {'answers': {'relevance': {'type': 'score', 'score': 7, 'probabilities': {'0': 1}}}}
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'test'}), patch('mori.providers.http_json', return_value=response):
            with self.assertRaises(ValueError):
                Jev(self.config, self.store).score('q', 't')

    def test_openai_embedding_contract_and_dimensions(self):
        config = load(ROOT / 'examples/openai.toml')
        config.embedding_dimensions = 3
        response = {'data': [{'index': 1, 'embedding': [0., 3., 4.]}, {'index': 0, 'embedding': [3., 4., 0.]}]}
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-not-a-secret'}), patch('mori.providers.http_json', return_value=response) as request:
            vectors = Embeddings(config).encode(['first', 'second'])
        self.assertEqual(request.call_args.args[0], 'https://api.openai.com/v1/embeddings')
        self.assertEqual(request.call_args.args[1]['model'], 'text-embedding-3-large')
        self.assertEqual(request.call_args.args[1]['dimensions'], 3)
        self.assertEqual(request.call_args.args[1]['encoding_format'], 'float')
        self.assertEqual(vectors[0], [0.6, 0.8, 0.0])

    def test_openai_missing_key_and_wrong_shape_fail(self):
        config = load(ROOT / 'examples/openai.toml')
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'API key'):
                Embeddings(config).encode(['text'])
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), patch('mori.providers.http_json', return_value={'data': [{'index': 0, 'embedding': [1., 0.]}]}):
            with self.assertRaisesRegex(ValueError, 'dimension count'):
                Embeddings(config).encode(['text'])

    def test_http_embeddings_reject_advertised_model_substitution(self):
        config = dataclasses.replace(self.config, embedding='http', embedding_model='explicit-model')
        response = {'model': 'another-model', 'data': [{'index': 0, 'embedding': [1.0, 0.0]}]}
        with patch('mori.providers.http_json', return_value=response):
            with self.assertRaisesRegex(ValueError, 'different model'):
                Embeddings(config).encode(['text'])
        response['model'] = config.embedding_model
        with patch('mori.providers.http_json', return_value=response):
            self.assertEqual(Embeddings(config).encode(['text']), [[1.0, 0.0]])

    def test_provider_redirects_never_forward_secrets_or_source_data(self):
        # Exercise urllib's real redirect handling with an in-memory transport.
        # No listener, provider connection or real credential is involved.
        original_builder = urllib.request.build_opener
        for code in (301, 302, 303, 307, 308):
            requests = []
            class RedirectingTransport(urllib.request.HTTPHandler):
                def http_open(self, request):
                    requests.append(request)
                    headers = Message()
                    headers['Location'] = 'http://unintended.invalid/leak'
                    response = urllib.response.addinfourl(io.BytesIO(b'withheld'), headers, request.full_url, code)
                    response.msg = 'Redirect'
                    return response
            with self.subTest(status=code), patch('mori.providers.urllib.request.build_opener',
                    side_effect=lambda *handlers: original_builder(*handlers, RedirectingTransport())):
                with self.assertRaisesRegex(RuntimeError, f'HTTP {code}'):
                    http_json('http://provider.invalid/v1', {'source': 'private fixture'}, 'test-only-key', 1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].full_url, 'http://provider.invalid/v1')

    def test_vector_normalization_handles_extreme_finite_values(self):
        for values in ([1e308, 1e308], [1e-308, -1e-308]):
            with self.subTest(values=values):
                normalized = vector(values)
                self.assertTrue(all(math.isfinite(value) for value in normalized))
                self.assertAlmostEqual(math.hypot(*normalized), 1.0)
                self.assertNotEqual(normalized, [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, 'Invalid embedding'):
            vector([10 ** 1000])

    def test_summary_segment_dimensions_fail_atomically(self):
        config = dataclasses.replace(self.config, chunk_words=2, chunk_overlap=0)
        embeddings = Mock(fingerprint={'model': 'dimension-fixture'})
        embeddings.encode.side_effect = lambda texts: [[1.0, 0.0] if i == 0 else [0.0, 0.0, 1.0] for i in range(len(texts))]
        store = Store(Path(self.temp.name) / 'dimensions')
        with self.assertRaisesRegex(ValueError, 'summary segment dimensions'):
            ingest(store, [{'id': 'new', 'source_id': 'work', 'title': 'Title',
                            'text': 'one two', 'summary': 'one two three four'}], config, embeddings)
        self.assertEqual(store.stats()['units'], 0)
        self.assertIsNone(store.fingerprint())

    def test_long_summaries_obey_embedding_batch_size(self):
        config = dataclasses.replace(self.config, chunk_words=2, chunk_overlap=0, embedding_batch=2)
        embeddings = Mock(fingerprint={'model': 'batch-fixture'})
        embeddings.encode.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        store = Store(Path(self.temp.name) / 'batches')
        ingest(store, [{'id': 'new', 'source_id': 'work', 'title': 'Title',
                        'text': 'one two', 'summary': 'one two three four five six seven eight nine ten'}], config, embeddings)
        self.assertEqual([len(call.args[0]) for call in embeddings.encode.call_args_list], [1, 2, 2, 1])

    def test_generated_summary_reimport_reuses_record_without_model_calls(self):
        unit = {**self.units[0], 'id': 'generated-summary', 'summary': ''}
        writer = Mock()
        writer.complete.return_value = 'Faithful test fixture summary of the complete unit.'
        ingest(self.store, [unit], self.config, self.embeddings, llm=writer, summarize=True)
        original = next(u for u in self.store.units() if u['id'] == unit['id'])
        with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected embedding call')):
            result = ingest(self.store, [unit], self.config, self.embeddings, summarize=True)
        self.assertEqual(result['unchanged_units'], 1)
        self.assertEqual(writer.complete.call_count, 1)
        self.assertEqual(next(u for u in self.store.units() if u['id'] == unit['id']), original)
        with self.assertRaisesRegex(ValueError, 'Changed unit ID'):
            ingest(self.store, [{**unit, 'text': 'A changed original'}], self.config, self.embeddings, llm=writer, summarize=True)
        self.assertEqual(writer.complete.call_count, 1)

    def test_invalid_later_unit_stops_before_summary_generation(self):
        writer = Mock()
        missing_summary = {**self.units[0], 'id': 'needs-summary', 'summary': ''}
        invalid = {**self.units[1], 'id': 'invalid id'}
        with self.assertRaises(ValueError):
            ingest(self.store, [missing_summary, invalid], self.config, self.embeddings, llm=writer, summarize=True)
        writer.complete.assert_not_called()

    def test_invalid_metadata_stops_before_any_provider_calls(self):
        writer = Mock()
        for metadata in ({'score': float('nan')}, {'nested': {'score': float('inf')}},
                         {1: 'numeric key'}, {'nested': {1: 'numeric key'}}, {'tuple': (1, 2)}):
            unit = {**self.units[0], 'id': 'bad-metadata', 'summary': '', 'metadata': metadata}
            with self.subTest(metadata=metadata), patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected embedding call')):
                with self.assertRaisesRegex(ValueError, 'finite JSON'):
                    ingest(self.store, [unit], self.config, self.embeddings, llm=writer, summarize=True)
        writer.complete.assert_not_called()

    def test_legacy_fingerprint_accepts_new_units_without_reembedding(self):
        fingerprint = self.store.fingerprint()
        fingerprint.pop('requested_dimensions')
        with self.store.connect() as db:
            db.execute("UPDATE meta SET value=? WHERE key='embedding'", (json.dumps(fingerprint),))
        result = ingest(self.store, [{**self.units[0], 'id': 'new-legacy-unit'}], self.config, self.embeddings)
        self.assertEqual(result['added_units'], 1)
        self.assertEqual(self.store.fingerprint(), fingerprint)

    def test_explicit_embedding_dimensions_reset_fails_before_provider(self):
        fingerprint = self.store.fingerprint()
        fingerprint['requested_dimensions'] = 256
        with self.store.connect() as db:
            db.execute("UPDATE meta SET value=? WHERE key='embedding'", (json.dumps(fingerprint),))
        with patch.object(self.embeddings, 'encode', side_effect=AssertionError('Unexpected embedding call')):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                ingest(self.store, [{**self.units[0], 'id': 'new-dimension-unit'}], self.config, self.embeddings)

    def test_docx_preserves_visible_spacing_and_omits_deleted_and_field_text(self):
        xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>first</w:t><w:tab/><w:t>second</w:t><w:br/><w:t>third</w:t></w:r></w:p><w:p><w:del><w:r><w:delText>deleted</w:delText></w:r></w:del><w:r><w:instrText> HYPERLINK hidden </w:instrText><w:t>visible</w:t></w:r></w:p></w:body></w:document>'
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('word/document.xml', xml)
        self.assertEqual(extract(data.getvalue(), '.docx'), 'first\tsecond\nthird\nvisible')

    def test_embedding_preset_initializer_preserves_existing_configs(self):
        from mori.cli import main
        output = Path(self.temp.name) / 'openai.toml'
        self.assertIsNone(main(['init', '--embeddings', 'openai', '--output', str(output)]))
        self.assertEqual(load(output).embedding_model, 'text-embedding-3-large')
        original = output.read_bytes()
        self.assertEqual(main(['init', '--embeddings', 'minilm', '--output', str(output)]), 1)
        self.assertEqual(output.read_bytes(), original)

    def test_500_to_100_to_20_mechanics(self):
        config = dataclasses.replace(self.config, chunk_words=8, chunk_overlap=0, full_units=1, summaries=5)
        store = Store(Path(self.temp.name) / 'large-fixture')
        units = []
        for i in range(12):
            text = ' '.join(f'water{i}_{j} catchment flows storage wetlands limits policy context' for j in range(50))
            units.append({'id': f'u{i}', 'source_id': f's{i}', 'title': f'Unit {i}', 'text': text, 'summary': f'Wetland storage unit {i}.'})
        embeddings = Embeddings(config)
        ingest(store, units, config, embeddings)
        writer = Writer()
        first = query(store, 'wetland storage', config, embeddings=embeddings, scorer=Scorer(), llm=writer)
        self.assertEqual((first['counts']['retrieved'], first['counts']['passages']), (500, 100))
        config.rerank_enabled = True
        second = query(store, 'wetland storage', config, embeddings=embeddings, scorer=Scorer(), llm=writer)
        self.assertEqual(second['counts']['passages'], 20)


if __name__ == '__main__':
    unittest.main()
