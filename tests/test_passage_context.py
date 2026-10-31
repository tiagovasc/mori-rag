"""Parent retrieval, exact evidence windows, and old/new packet rendering."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from mori.config import Config
from mori.pipeline import passage_context, query, rating_text, render_context, retrieval_similarity, validate_answer
from mori.providers import expected_embedding_fingerprint
from mori.store import digest
from test_frontend import HARNESS, NODE


ROOT = Path(__file__).resolve().parents[1]
CHILD_POLICY = 'max-child-cosine-v1'


class QueryEmbeddings:
    fingerprint = {'model': 'parent-retrieval-test'}

    def encode(self, texts, query=False):
        return [[1.0, 0.0] for _ in texts]


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def score(self, question, text, purpose):
        self.calls.append((purpose, json.loads(text)))
        return {'score': 80.0}


class MemoryStore:
    def __init__(self, root, units, rows, child_policy=True):
        self.root, self._units, self._rows = Path(root), units, rows
        self.policy = expected_embedding_fingerprint(Config(embedding='demo', chunk_strategy='sentence' if child_policy else 'words'),
                                                     QueryEmbeddings())

    def fingerprint(self):
        return dict(self.policy)

    def stats(self):
        return {'units': len(self._units), 'chunks': len(self._rows), 'embedding': self.fingerprint()}

    def units(self):
        return copy.deepcopy(self._units)

    def chunks(self):
        return copy.deepcopy(self._rows)


def unit(identifier, text):
    return {'id': identifier, 'source_id': 'source-' + identifier, 'title': 'Title ' + identifier,
            'kind': 'chapter', 'locator': 'Chapter ' + identifier, 'metadata': {'author': 'Test author'},
            'hash': digest(text), 'text': text, 'summary': 'Summary of ' + identifier,
            'summary_vector': [0.0, 1.0]}


def parent(source, identifier, start=0, end=None, seq=0, children=None):
    end = len(source['text']) if end is None else end
    row = {'id': identifier, 'unit_id': source['id'], 'text': source['text'][start:end],
           'start': start, 'end': end, 'seq': seq, 'vector': [0.0, 1.0]}
    if children is not None:
        middle = start + (end - start) // 2
        row.update(search_vectors=children,
                   embedding_spans=[{'seq': index, 'start': start if index == 0 else middle,
                                     'end': middle if index == 0 and len(children) > 1 else end}
                                    for index in range(len(children))])
    return row


class PassageContextTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.local' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.a = unit('a', 'The first complete argument. Its qualification matters.')
        self.b = unit('b', 'Another complete source explains a different argument.')
        self.rows = [parent(self.a, 'pa', children=[[0.0, 1.0], [1.0, 0.0]]),
                     parent(self.b, 'pb', children=[[0.8, 0.6]])]
        self.store = MemoryStore(self.temp.name, [self.a, self.b], self.rows)
        self.config = Config(embedding='demo', retrieval_k=10, jev_k=10, final_passages=10,
                             full_units=0, summaries=0, min_relevance=0, summary_min_relevance=0).validate()
        self.scorer = RecordingScorer()

    def run_query(self):
        result = query(self.store, 'What is the complete argument?', self.config,
                       embeddings=QueryEmbeddings(), scorer=self.scorer)
        self.run = Path(result['directory'])
        return json.loads((self.run / 'context.json').read_text(encoding='utf-8'))

    def test_max_child_retrieves_parent_once_and_preserves_winning_span(self):
        packet = self.run_query()
        self.assertEqual([row['id'] for row in packet['passages']], ['pa', 'pb'])
        self.assertEqual(packet['counts']['retrieved'], 2)
        self.assertEqual(packet['counts']['rated_passages'], 2)
        winner = packet['passages'][0]
        self.assertEqual(winner['retrieval_match'], {**self.rows[0]['embedding_spans'][1], 'score': 1.0})
        self.assertEqual(winner['target']['text'], self.a['text'])
        self.assertEqual(winner['context']['text'], self.a['text'])
        self.assertEqual(self.scorer.calls[0][1]['context']['text'], self.a['text'])
        self.assertEqual(len(self.scorer.calls), 2)
        validate_answer('The first argument has an explicit qualification. [P1]', packet)

    def test_child_score_ties_choose_lowest_sequence_independent_of_array_order(self):
        row = parent(self.a, 'pa', children=[[1.0, 0.0], [1.0, 0.0]])
        row['embedding_spans'].reverse()
        score, match = retrieval_similarity(row, [1.0, 0.0])
        self.assertEqual(score, 1.0)
        self.assertEqual(match['seq'], 0)

    def test_saved_artifacts_contain_lineage_but_no_vectors(self):
        self.run_query()
        forbidden = {'vector', 'search_vectors', 'summary_vector', 'summary_search_vectors'}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        for name in ('retrieved.json', 'passage_ratings.json', 'summary_ratings.json', 'context.json', 'request.json'):
            with self.subTest(name=name):
                inspect(json.loads((self.run / name).read_text(encoding='utf-8')))
        retrieved = json.loads((self.run / 'retrieved.json').read_text(encoding='utf-8'))
        self.assertEqual(retrieved[0]['retrieval_match']['seq'], 1)
        self.assertEqual(retrieved[0]['embedding_spans'], self.rows[0]['embedding_spans'])

    def test_empty_and_missing_new_children_fail_without_flat_vector_fallback(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                row = copy.deepcopy(self.rows[0])
                if missing:
                    row.pop('search_vectors')
                else:
                    row['search_vectors'] = []
                with self.assertRaisesRegex(ValueError, 'child embeddings'):
                    retrieval_similarity(row, [1.0, 0.0], require_children=True)
        self.rows[0]['search_vectors'] = []
        with self.assertRaisesRegex(ValueError, 'no child embeddings'):
            self.run_query()
        self.assertEqual(self.scorer.calls, [])

    def test_child_span_mismatch_and_out_of_parent_offsets_fail(self):
        for spans in ([], [{'seq': 0, 'start': -1, 'end': 10}],
                      [{'seq': 0, 'start': 0, 'end': len(self.a['text']) + 1}]):
            row = parent(self.a, 'pa', children=[[1.0, 0.0]])
            row['embedding_spans'] = spans
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, 'spans'):
                retrieval_similarity(row, [1.0, 0.0])

    def test_partial_child_coverage_and_missing_sequence_fail_closed(self):
        row = parent(self.a, 'pa', children=[[1.0, 0.0], [0.0, 1.0]])
        broken = [
            [{'seq': 0, 'start': 4, 'end': len(self.a['text'])}],
            [{'seq': 0, 'start': 0, 'end': 10}],
            [{'seq': 0, 'start': 0, 'end': 10}, {'seq': 1, 'start': 20, 'end': len(self.a['text'])}],
            [{'seq': 0, 'start': 0, 'end': 25}, {'seq': 2, 'start': 25, 'end': len(self.a['text'])}],
        ]
        for spans in broken:
            damaged = {**row, 'embedding_spans': spans, 'search_vectors': row['search_vectors'][:len(spans)]}
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, 'cover|sequence'):
                retrieval_similarity(damaged, [1.0, 0.0])

    def test_whitespace_between_complete_children_is_not_lost_source_content(self):
        source = unit('a', 'First.\r\n\r\nLast.')
        row = parent(source, 'pa', children=[[1.0, 0.0], [0.0, 1.0]])
        row['embedding_spans'] = [{'seq': 0, 'start': 0, 'end': 6},
                                  {'seq': 1, 'start': 10, 'end': len(source['text'])}]
        self.assertEqual(retrieval_similarity(row, [1.0, 0.0])[0], 1.0)

    def test_nonfinite_or_wrong_dimension_child_vectors_fail(self):
        for invalid in ([float('nan'), 0.0], [float('inf'), 0.0], [True, 0.0], [1.0]):
            row = parent(self.a, 'pa', children=[invalid])
            with self.subTest(vector=invalid), self.assertRaisesRegex(ValueError, 'vector|dimensions'):
                retrieval_similarity(row, [1.0, 0.0])

    def test_zero_and_unnormalized_children_fail_before_relevance_calls(self):
        for invalid in ([0.0, 0.0], [2.0, 0.0], [0.1, 0.1]):
            row = parent(self.a, 'pa', children=[invalid])
            with self.subTest(vector=invalid), self.assertRaisesRegex(ValueError, 'unit norm'):
                retrieval_similarity(row, [1.0, 0.0])
        self.rows[0]['search_vectors'][1] = [2.0, 0.0]
        with self.assertRaisesRegex(ValueError, 'unit norm'):
            self.run_query()
        self.assertEqual(self.scorer.calls, [])

    def test_legacy_flat_index_still_queries_without_new_fields(self):
        self.store = MemoryStore(self.temp.name, [self.a], [parent(self.a, 'legacy')], child_policy=False)
        self.config.chunk_strategy = 'words'
        packet = self.run_query()
        self.assertEqual(packet['passages'][0]['id'], 'legacy')
        self.assertNotIn('retrieval_match', packet['passages'][0])
        self.assertEqual(packet['passages'][0]['context']['text'], self.a['text'])

    def test_new_children_require_explicit_fingerprinted_pooling(self):
        self.store.policy.pop('retrieval_pooling')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_query()
        self.assertEqual(self.scorer.calls, [])

    def test_switching_between_legacy_and_child_policy_fails_before_query_embedding(self):
        provider = QueryEmbeddings()

        def unexpected(*args, **kwargs):
            self.fail('A mismatched index must fail before query embedding')

        provider.encode = unexpected
        for index_children, requested_strategy in ((True, 'words'), (False, 'sentence')):
            self.store = MemoryStore(self.temp.name, [self.a], [parent(self.a, 'pa')], child_policy=index_children)
            self.config.chunk_strategy = requested_strategy
            with self.subTest(index_children=index_children), self.assertRaisesRegex(ValueError, 'fingerprint'):
                query(self.store, 'Question', self.config, embeddings=provider, scorer=self.scorer)

    def test_summary_preselection_uses_max_child_and_records_match(self):
        for source, vectors in ((self.a, [[0.0, 1.0], [1.0, 0.0]]), (self.b, [[0.8, 0.6]])):
            source['summary_search_vectors'] = vectors
            length = len(source['summary'])
            source['summary_embedding_spans'] = [{'seq': index, 'start': 0 if index == 0 else length // 2,
                                                 'end': length // 2 if index == 0 and len(vectors) > 1 else length}
                                                for index in range(len(vectors))]
        self.config.full_units = 1
        self.config.summary_candidates = 1
        packet = self.run_query()
        self.assertEqual(packet['full_units'][0]['id'], 'a')
        summaries = [data for kind, data in self.scorer.calls if kind == 'complete source summary']
        self.assertEqual(summaries, [{'title': self.a['title'], 'summary': self.a['summary']}])
        saved = json.loads((self.run / 'summary_ratings.json').read_text(encoding='utf-8'))
        self.assertEqual(saved[0]['retrieval_match']['seq'], 1)
        self.assertNotIn('summary_search_vectors', packet['full_units'][0])

    def test_summary_children_without_declared_policy_fail(self):
        self.a['summary_search_vectors'] = [[1.0, 0.0]]
        self.a['summary_embedding_spans'] = [{'seq': 0, 'start': 0, 'end': len(self.a['summary'])}]
        self.store.policy['summary_pooling'] = 'legacy-mean'
        self.config.full_units = 1
        self.config.summary_candidates = 1
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_query()

    def test_overlapping_neighbors_render_once_in_original_order_and_within_unit(self):
        text = 'First sentence.\r\n\r\nSecond sentence.\u2029Third sentence.'
        source = unit('a', text)
        first_end = text.index('Second') + len('Second sentence.')
        middle_start, middle_end = text.index('Second'), text.index('Third')
        rows = [parent(source, 'left', 0, first_end, 0),
                parent(source, 'middle', middle_start, middle_end, 1),
                parent(source, 'right', middle_start, len(text), 2)]
        by_unit = {'a': list(reversed(rows)), 'b': [parent(self.b, 'other')]}
        context = passage_context(rows[1], source, by_unit, 1)
        self.assertEqual(context, {'text': text, 'start': 0, 'end': len(text)})
        state = json.loads(rating_text(rows[1], source, by_unit, 1))
        self.assertEqual(state['context'], context)
        self.assertEqual(state['target'], {'start': middle_start, 'end': middle_end})
        self.assertEqual(state['context']['text'].count('Second sentence.'), 1)
        self.assertNotIn(self.b['text'], json.dumps(state))
        self.assertTrue(all('text' not in row for row in state['surrounding_context']))

    def test_corrupt_original_offsets_are_rejected(self):
        row = parent(self.a, 'pa')
        row['text'] = 'A substituted source.'
        with self.assertRaisesRegex(ValueError, 'offsets do not match'):
            passage_context(row, self.a, {'a': [row]}, 0)

    def test_identical_expanded_windows_are_selected_once_and_citation_is_valid(self):
        self.store = MemoryStore(self.temp.name, [self.a], [parent(self.a, 'first', 0, 30, 0),
                                                         parent(self.a, 'second', 20, len(self.a['text']), 1)],
                                 child_policy=False)
        self.config.chunk_strategy = 'words'
        self.config.answer_neighbors = 1
        packet = self.run_query()
        self.assertEqual(len(packet['passages']), 1)
        self.assertEqual(packet['passages'][0]['context']['text'], self.a['text'])
        self.assertTrue(any('1 available' in notice for notice in packet['notices']))
        rendered = render_context(packet)
        self.assertEqual(rendered.count(self.a['text']), 1)
        self.assertNotIn('Adjacent context', rendered)
        validate_answer('This answer uses the complete retrieved source span. [P1]', packet)

    def test_legacy_packet_without_context_remains_renderable(self):
        packet = self.run_query()
        for passage in packet['passages']:
            passage.pop('context')
        rendered = render_context(packet)
        self.assertIn(self.a['text'], rendered)
        self.assertIn(self.b['text'], rendered)
        self.assertIn('[P1]', rendered)


@unittest.skipUnless(NODE, 'Node.js is required for frontend behavior checks')
class PassageRenderingTests(unittest.TestCase):
    def run_ui(self, scenario):
        code = HARNESS + '\n(async()=>{\n' + scenario + '\n})().catch(error=>{console.error(error);process.exitCode=1;});'
        result = subprocess.run([NODE, '-e', code], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ui_displays_exact_context_once_with_size_and_citation_identity(self):
        self.run_ui(r"""
const app=browser();await settle();
const saved=packet('new inquiry',[alpha]);
saved.context.passages=[{citation:'P1',unit:{title:'Source title',locator:'Chapter 1',source_id:'a'},rating:{score:80},
target:{text:'Middle words.',start:7,end:20},neighbors:[{text:'Before Middle words.'}],
context:{text:'Before Middle words. After <script>literal</script>.',start:0,end:51}}];
saved.context.counts.passages=1;saved.answer='Supported statement. [P1]';app.server.runReplies.new=saved;
await app.openRun('new');
const evidence=app.el('evidence-view').children[0];
assert.equal(evidence.dataset.citation,'P1');assert.equal(evidence.children[2].textContent,saved.context.passages[0].context.text);
assert.equal(evidence.children[2].textContent.split('Middle words.').length-1,1);
assert.match(evidence.children[1].textContent,/2-word passage.*5 words with context/);
assert.ok(app.el('answer-view').children[0].children.some(child=>child.textContent==='[P1]'));
""")

    def test_ui_keeps_legacy_saved_passages_readable(self):
        self.run_ui(r"""
const app=browser();await settle();const saved=packet('old inquiry');
saved.context.passages=[{citation:'P1',unit:{title:'Old title',locator:'Old chapter',source_id:'old'},rating:{score:75},
target:{text:'Original stored target.',start:0,end:23},neighbors:[{text:'Stored adjacent evidence.'}]}];
saved.context.counts.passages=1;app.server.runReplies.old=saved;await app.openRun('old');
const evidence=app.el('evidence-view').children[0];
assert.equal(evidence.children[2].textContent,'Original stored target.\n\nAdjacent context:\nStored adjacent evidence.');
assert.match(evidence.children[1].textContent,/3-word passage/);
""")


if __name__ == '__main__':
    unittest.main()
