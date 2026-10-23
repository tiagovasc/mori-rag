"""ONNX tokenizer contract, without requiring optional model dependencies in CI."""
import math
import types
import unittest
from unittest.mock import patch

from mori.config import Config
from mori.ingest import ingest
from mori.providers import Embeddings


class OnnxContractTests(unittest.TestCase):
    def backend(self):
        state = types.SimpleNamespace(disabled=False, model_runs=0, token_counts=[])

        class Tokenizer:
            @classmethod
            def from_file(cls, path):
                return cls()

            def no_truncation(self):
                state.disabled = True

            def enable_padding(self):
                pass

            def encode_batch(self, texts):
                result = []
                for text in texts:
                    ids = list(range(len(text.split())))
                    # Simulate a tokenizer.json that silently truncates to2.
                    if not state.disabled:
                        ids = ids[:2]
                    state.token_counts.append(len(ids))
                    result.append(types.SimpleNamespace(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids)))
                return result

        class Session:
            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [types.SimpleNamespace(name='input_ids')]

            def get_outputs(self):
                return [types.SimpleNamespace(name='sentence_embedding')]

            def run(self, names, inputs):
                state.model_runs += 1
                return [types.SimpleNamespace(tolist=lambda: [[1.0, 2.0] for _ in inputs['input_ids']])]

        modules = {
            'tokenizers': types.SimpleNamespace(Tokenizer=Tokenizer),
            'onnxruntime': types.SimpleNamespace(SessionOptions=types.SimpleNamespace, InferenceSession=Session),
            'numpy': types.SimpleNamespace(asarray=lambda value, dtype: value, int64=int),
        }
        return state, modules

    def test_inherited_truncation_is_disabled_and_oversized_input_fails_before_inference(self):
        state, modules = self.backend()
        config = Config(embedding='onnx', embedding_token_limit=3, onnx_pooling='sentence_embedding')
        with patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(ValueError, 'no ONNX input was truncated'):
                Embeddings(config).encode(['one two three four'])
        self.assertTrue(state.disabled)
        self.assertEqual(state.token_counts, [4])
        self.assertEqual(state.model_runs, 0)

    def test_accepted_input_reaches_inference_without_inherited_truncation(self):
        state, modules = self.backend()
        config = Config(embedding='onnx', embedding_token_limit=3, onnx_pooling='sentence_embedding')
        with patch.dict('sys.modules', modules):
            vectors = Embeddings(config).encode(['one two three'])
        self.assertTrue(state.disabled)
        self.assertEqual(state.token_counts, [3])
        self.assertEqual(state.model_runs, 1)
        self.assertEqual(len(vectors), 1)
        self.assertAlmostEqual(math.hypot(*vectors[0]), 1.0)

    def test_tokenization_version_changes_only_onnx_fingerprints(self):
        onnx = Embeddings(Config(embedding='onnx')).fingerprint
        self.assertEqual(onnx['onnx_tokenization'], 'untruncated-v1')
        legacy_onnx = {key: value for key, value in onnx.items() if key != 'onnx_tokenization'}
        legacy_store = types.SimpleNamespace(fingerprint=lambda: legacy_onnx)
        with self.assertRaisesRegex(ValueError, 'Embedding fingerprint mismatch before import'):
            ingest(legacy_store, [], Config(embedding='onnx'), Embeddings(Config(embedding='onnx')))
        for name in ('demo', 'local', 'http'):
            config = Config(embedding=name)
            expected = {
                'provider': name, 'model': config.embedding_model, 'revision': config.embedding_revision,
                'query_prefix': config.query_prefix, 'document_prefix': config.document_prefix,
                'normalization': 'l2', 'endpoint': config.embedding_url if name == 'http' else '',
                'onnx_model_file': '', 'pooling': '', 'token_limit': config.embedding_token_limit,
                'requested_dimensions': config.embedding_dimensions,
                'chunk_words': config.chunk_words, 'chunk_overlap': config.chunk_overlap,
            }
            self.assertEqual(Embeddings(config).fingerprint, expected)


if __name__ == '__main__':
    unittest.main()
