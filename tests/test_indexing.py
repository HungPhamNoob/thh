import argparse
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vibiomir.chunking import build_chunker
from vibiomir.indexing.numpy import build_numpy_index
from vibiomir.indexing.sparse import export_impact_vectors
from vibiomir.indexing.store import ArtifactStore, prepare_store
from vibiomir.retrieval.numpy_index import NumpyIndexRetriever
from vibiomir.schemas import Document


class ArtifactStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store_path = self.root / "store.sqlite"
        documents = [
            Document(id="vi-1", text="đau đầu kéo dài", language="vi"),
            Document(id="en-1", text="persistent headache treatment", language="en"),
        ]
        chunker = build_chunker({"type": "fixed", "size": 20, "overlap": 0, "min_tokens": 1})
        prepare_store(documents, chunker, self.store_path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_store_preserves_exact_chunks_and_language_filter(self):
        with ArtifactStore(self.store_path) as store:
            self.assertEqual(store.count("document"), 2)
            self.assertEqual(store.count("chunk", ["en"]), 1)
            item = next(store.iter_items("chunk", languages=["vi"]))[0]
            chunk = store.get_chunks([item.id])[item.id]
            self.assertEqual(chunk.text, "đau đầu kéo dài")
            self.assertEqual(chunk.text, chunk.index_text)

    def test_store_v3_keeps_text_once_and_preserves_index_override(self):
        from vibiomir.chunking.base import make_chunk

        class OverrideChunker:
            def split(self, document):
                chunk = make_chunk(document, 0, 0, len(document.text))
                chunk.index_text = f"CONTEXT {chunk.text}"
                return [chunk]

        override_path = self.root / "override.sqlite"
        prepare_store(
            [Document(id="d1", text="hello world", language="en")],
            OverrideChunker(),
            override_path,
        )
        with ArtifactStore(override_path) as store:
            columns = {
                row["name"]
                for row in store.connection.execute("PRAGMA table_info(chunks)")
            }
            # Chunk text is derived from documents.text; only a differing
            # index_text is stored, never a second copy of the chunk text.
            self.assertNotIn("text", columns)
            self.assertIn("index_text_override", columns)
            stored = store.connection.execute(
                "SELECT index_text_override FROM chunks"
            ).fetchone()["index_text_override"]
            self.assertEqual(stored, "CONTEXT hello world")
            chunk = store.get_chunks(["d1::c00000"])["d1::c00000"]
            self.assertEqual(chunk.text, "hello world")
            self.assertEqual(chunk.index_text, "CONTEXT hello world")

        with ArtifactStore(self.store_path) as store:
            stored = store.connection.execute(
                "SELECT index_text_override FROM chunks"
            ).fetchall()
            self.assertTrue(all(row["index_text_override"] is None for row in stored))
            items = next(store.iter_items("chunk", batch_size=10))
            self.assertEqual({item.text for item in items if item.id.startswith("vi")},
                             {"đau đầu kéo dài"})
            metadata = store.metadata()
            self.assertEqual(metadata.get("store_format_version"), 3)

    def test_quantization_spec_maps_to_scalar_int8(self):
        from qdrant_client import models

        from vibiomir.indexing.qdrant import quantization_config_from_spec

        config = quantization_config_from_spec(
            {"scalar": {"quantile": 0.95, "always_ram": True}}
        )
        assert config is not None
        self.assertEqual(config.scalar.type, models.ScalarType.INT8)
        self.assertEqual(config.scalar.quantile, 0.95)
        self.assertTrue(config.scalar.always_ram)
        self.assertIsNone(quantization_config_from_spec(None))
        self.assertIsNone(quantization_config_from_spec({}))
        with self.assertRaises(ValueError):
            quantization_config_from_spec({"binary": {}})

    def test_store_exposes_reproducible_chunk_provenance(self):
        with ArtifactStore(self.store_path) as store:
            row = store.connection.execute(
                "SELECT doc_id, chunk_id, chunk_order, text, index_text, start_char, "
                "end_char, language, source_url FROM chunk_provenance ORDER BY chunk_id LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["text"], row["index_text"])
            self.assertEqual(row["text"], "persistent headache treatment")

    def test_export_jsonl_gzip_roundtrip(self):
        output = self.root / "collection" / "part-00000.json.gz"
        with ArtifactStore(self.store_path) as store:
            self.assertEqual(store.export_jsonl("chunk", output), 2)
        rows = [json.loads(line) for line in gzip.open(output, "rt", encoding="utf-8")]
        self.assertEqual(len(rows), 2)
        self.assertIn("contents", rows[0])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "ingest extra not installed")
    def test_prepare_accepts_multi_shard_parquet_and_checks_counts(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vibiomir.indexing.cli import command_prepare

        corpus = self.root / "corpus"
        corpus.mkdir()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"id": 1, "text": "nội dung một", "language": "vi"},
                    {"id": 2, "text": "", "language": "vi"},
                ]
            ),
            corpus / "part-00000.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [{"id": 3, "text": "document three", "language": "en"}]
            ),
            corpus / "part-00001.parquet",
        )
        output = self.root / "multi.sqlite"
        args = argparse.Namespace(
            config="configs/smoke/qdrant_hash.yaml",
            set=[],
            corpus=str(corpus),
            hf_dataset=None,
            hf_file=None,
            hf_pattern=None,
            hf_revision=None,
            hf_max_workers=1,
            expected_shards=2,
            expected_source_rows=3,
            expected_documents=2,
            store=str(output),
        )
        self.assertEqual(command_prepare(args), 0)
        with ArtifactStore(output) as store:
            self.assertEqual(store.count("document"), 2)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "ingest extra not installed")
    def test_hf_snapshot_uses_only_matching_corpus_shards(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vibiomir.indexing.cli import command_prepare

        snapshot = self.root / "snapshot"
        corpus = snapshot / "corpus" / "v1"
        corpus.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist([{"id": 1, "text": "one", "language": "en"}]),
            corpus / "part-00000.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([{"id": 2, "text": "two", "language": "en"}]),
            corpus / "part-00001.parquet",
        )
        # This file may already exist in the shared HF snapshot cache.  It must
        # not leak into the explicitly selected corpus/v1 pattern.
        pq.write_table(
            pa.Table.from_pylist([{"id": 999, "url": "https://example.org"}]),
            snapshot / "manifest.parquet",
        )
        output = self.root / "hf.sqlite"
        args = argparse.Namespace(
            config="configs/smoke/qdrant_hash.yaml",
            set=[],
            corpus=None,
            hf_dataset="owner/dataset",
            hf_file=None,
            hf_pattern=["corpus/v1/*.parquet"],
            hf_revision="deadbeef",
            hf_max_workers=1,
            expected_shards=2,
            expected_source_rows=2,
            expected_documents=2,
            store=str(output),
        )
        with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
            self.assertEqual(command_prepare(args), 0)
        with ArtifactStore(output) as store:
            self.assertEqual(store.count("document"), 2)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "ingest extra not installed")
    def test_official_hf_contract_requires_exact_thirteen_shards(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vibiomir.indexing import cli

        snapshot = self.root / cli.VIBIOMIR_CORPUS_REVISION
        corpus = snapshot / "corpus" / "v1"
        corpus.mkdir(parents=True)
        for index in range(13):
            pq.write_table(
                pa.Table.from_pylist(
                    [{"id": index + 1, "text": "text", "language": "vi"}]
                ),
                corpus / f"part-{index:05d}.parquet",
            )
        pq.write_table(
            pa.Table.from_pylist([{"id": 999, "url": "https://example.org"}]),
            snapshot / "manifest.parquet",
        )
        args = argparse.Namespace(
            config="configs/smoke/qdrant_hash.yaml",
            set=[],
            corpus=None,
            hf_dataset=cli.VIBIOMIR_CORPUS_DATASET,
            hf_file=None,
            hf_pattern=[cli.VIBIOMIR_CORPUS_PATTERN],
            hf_revision=cli.VIBIOMIR_CORPUS_REVISION,
            hf_max_workers=1,
            expected_shards=None,
            expected_source_rows=None,
            expected_documents=None,
            expected_empty_rows=None,
            required_text_field="text",
            official_vibiomir_corpus=True,
            store=str(self.root / "official.sqlite"),
        )
        with (
            patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
            patch.object(cli, "VIBIOMIR_SOURCE_ROWS", 13),
            patch.object(cli, "VIBIOMIR_DOCUMENTS", 13),
            patch.object(cli, "VIBIOMIR_EMPTY_ROWS", 0),
        ):
            self.assertEqual(cli.command_prepare(args), 0)

    def test_official_contract_counts_are_content_consistent(self):
        from vibiomir.indexing import cli

        # corpus_info.json reports 1,047,415 non-empty / 173,183 empty by source
        # label; 31 rows labelled crawl/wayback carry empty text at the pinned
        # revision, so the content-based contract is exactly 31 lower/higher and
        # must still sum to the source row count.
        self.assertEqual(
            cli.VIBIOMIR_DOCUMENTS + cli.VIBIOMIR_EMPTY_ROWS, cli.VIBIOMIR_SOURCE_ROWS
        )
        self.assertEqual(cli.VIBIOMIR_DOCUMENTS, 1_047_415 - 31)
        self.assertEqual(cli.VIBIOMIR_EMPTY_ROWS, 173_183 + 31)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "ingest extra not installed")
    def test_missing_text_schema_fails_instead_of_treating_rows_as_empty(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vibiomir.indexing.cli import command_prepare

        corpus = self.root / "missing-text.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": 1, "language": "vi"}]), corpus)
        args = argparse.Namespace(
            config="configs/smoke/qdrant_hash.yaml",
            set=[],
            corpus=str(corpus),
            hf_dataset=None,
            hf_file=None,
            hf_pattern=None,
            hf_revision=None,
            hf_max_workers=1,
            expected_shards=1,
            expected_source_rows=1,
            expected_documents=0,
            expected_empty_rows=1,
            required_text_field="text",
            official_vibiomir_corpus=False,
            store=str(self.root / "invalid.sqlite"),
        )
        with self.assertRaisesRegex(ValueError, "required field 'text' is absent"):
            command_prepare(args)

    def test_persistent_numpy_index_searches_without_loading_corpus(self):
        index_dir = self.root / "index"
        build_numpy_index(
            self.store_path,
            "chunk",
            {"type": "hash", "dimension": 32, "normalize": True},
            index_dir,
        )
        retriever = NumpyIndexRetriever(str(index_dir))
        retriever.fit([])
        results = retriever.search("headache treatment", 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].item_id.startswith("en-1:"))

    def test_impact_export_resumes_after_a_committed_batch(self):
        output = self.root / "impact.jsonl"

        class InterruptingEncoder:
            calls = 0

            def encode(self, texts):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated interruption")
                return [{text: 100} for text in texts]

        with patch(
            "vibiomir.indexing.sparse.build_sparse_encoder",
            return_value=InterruptingEncoder(),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                export_impact_vectors(
                    self.store_path,
                    "chunk",
                    {"type": "test"},
                    output,
                    batch_size=1,
                )

        class WorkingEncoder:
            def encode(self, texts):
                return [{text: 100} for text in texts]

        with patch(
            "vibiomir.indexing.sparse.build_sparse_encoder",
            return_value=WorkingEncoder(),
        ):
            count = export_impact_vectors(
                self.store_path,
                "chunk",
                {"type": "test"},
                output,
                batch_size=1,
            )
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(count, 2)
        self.assertEqual(len({row["id"] for row in rows}), 2)
        self.assertFalse(output.with_suffix(".jsonl.state.json").exists())

    def test_qdrant_build_resumes_and_keeps_exact_chunk_payload(self):
        from qdrant_client import QdrantClient

        from vibiomir.indexing.qdrant import build_qdrant_index

        state_dir = self.root / "qdrant-state"
        local_path = self.root / "qdrant-data"

        class InterruptingEncoder:
            calls = 0

            def encode_documents(self, texts):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated interruption")
                return np.ones((len(texts), 4), dtype=np.float32)

        with patch(
            "vibiomir.indexing.qdrant.build_text_encoder",
            return_value=InterruptingEncoder(),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                build_qdrant_index(
                    self.store_path,
                    "chunk",
                    {"type": "test"},
                    "test_chunks",
                    state_dir=state_dir,
                    batch_size=1,
                    local_path=local_path,
                    on_disk=False,
                )

        class WorkingEncoder:
            def encode_documents(self, texts):
                return np.ones((len(texts), 4), dtype=np.float32)

        with patch("vibiomir.indexing.qdrant.build_text_encoder", return_value=WorkingEncoder()):
            manifest = build_qdrant_index(
                self.store_path,
                "chunk",
                {"type": "test"},
                "test_chunks",
                state_dir=state_dir,
                batch_size=1,
                local_path=local_path,
                on_disk=False,
            )
        self.assertEqual(manifest["items"], 2)
        self.assertFalse(state_dir.joinpath("build_state.json").exists())
        client = QdrantClient(path=str(local_path))
        try:
            points, _ = client.scroll(
                collection_name="test_chunks", limit=10, with_payload=True
            )
            self.assertEqual(len(points), 2)
            payloads = {point.payload["item_id"]: point.payload for point in points}
            self.assertEqual(payloads["vi-1::c00000"]["chunk_text"], "đau đầu kéo dài")
            self.assertEqual(payloads["en-1::c00000"]["doc_id"], "en-1")
        finally:
            client.close()

    def test_qdrant_retriever_searches_local_collection(self):
        from vibiomir.indexing.qdrant import build_qdrant_index
        from vibiomir.retrieval.qdrant_index import QdrantIndexRetriever

        state_dir = self.root / "search-state"
        local_path = self.root / "search-qdrant"
        encoder = {"type": "hash", "dimension": 32, "normalize": True}
        build_qdrant_index(
            self.store_path,
            "chunk",
            encoder,
            "search_chunks",
            state_dir=state_dir,
            local_path=local_path,
            on_disk=False,
        )
        retriever = QdrantIndexRetriever(
            collection_name="search_chunks",
            encoder=encoder,
            local_path=local_path,
        )
        try:
            retriever.fit([])
            results = retriever.search("headache treatment", 1)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].item_id.startswith("en-1:"))
        finally:
            retriever.close()

    def _parallel_store(self):
        """Ten single-chunk documents for parallel-upload tests."""
        path = self.root / "parallel-store.sqlite"
        documents = [
            Document(id=f"vi-{index}", text=f"nội dung tài liệu {index}", language="vi")
            for index in range(1, 11)
        ]
        chunker = build_chunker({"type": "fixed", "size": 20, "overlap": 0, "min_tokens": 1})
        prepare_store(documents, chunker, path)
        return path

    def test_qdrant_parallel_upload_upserts_every_point_once(self):
        from vibiomir.indexing import qdrant as qdrant_module

        store_path = self._parallel_store()
        client = _FakeQdrantClient(jitter=True)
        encoder = _ConstantEncoder()
        with (
            patch.object(qdrant_module, "build_text_encoder", return_value=encoder),
            patch.object(qdrant_module, "make_qdrant_client", return_value=client),
        ):
            manifest = qdrant_module.build_qdrant_index(
                store_path,
                "chunk",
                {"type": "test"},
                "parallel_chunks",
                state_dir=self.root / "parallel-state",
                batch_size=1,
                upload_workers=4,
                on_disk=False,
            )
        self.assertEqual(manifest["items"], 10)
        self.assertEqual(manifest["verification"]["point_count"], 10)
        self.assertEqual(len(client.points), 10)
        self.assertEqual(client.calls, 10)
        self.assertFalse((self.root / "parallel-state" / "build_state.json").exists())

    def test_qdrant_parallel_upload_checkpoints_contiguously_and_resumes(self):
        from vibiomir.indexing import qdrant as qdrant_module

        store_path = self._parallel_store()
        with ArtifactStore(store_path) as store:
            item_ids = [item.id for item in next(store.iter_items("chunk", batch_size=100))]
        poison = item_ids[len(item_ids) // 2]
        client = _FakeQdrantClient(jitter=True, fail_ids={poison})
        state_dir = self.root / "resume-state"
        state_path = state_dir / "build_state.json"
        with (
            patch.object(qdrant_module, "build_text_encoder", return_value=_ConstantEncoder()),
            patch.object(qdrant_module, "make_qdrant_client", return_value=client),
            patch.object(qdrant_module.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated upsert failure"):
                qdrant_module.build_qdrant_index(
                    store_path,
                    "chunk",
                    {"type": "test"},
                    "parallel_chunks",
                    state_dir=state_dir,
                    batch_size=1,
                    upload_workers=4,
                    on_disk=False,
                )
            self.assertTrue(state_path.exists())
            checkpoint = json.loads(state_path.read_text())
            self.assertLess(checkpoint["offset"], 10)
            # The poisoned batch must not be acknowledged into the offset.
            self.assertLessEqual(checkpoint["offset"], item_ids.index(poison))

            client.fail_ids.clear()
            manifest = qdrant_module.build_qdrant_index(
                store_path,
                "chunk",
                {"type": "test"},
                "parallel_chunks",
                state_dir=state_dir,
                batch_size=1,
                upload_workers=4,
                on_disk=False,
            )
        self.assertEqual(manifest["verification"]["point_count"], 10)
        self.assertEqual(len(client.points), 10)  # deterministic ids: no duplicates
        self.assertFalse(state_path.exists())


class _ConstantEncoder:
    def encode_documents(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeQdrantClient:
    """Thread-safe stand-in exercising the parallel upload path without a server."""

    def __init__(self, jitter: bool = False, fail_ids=()):
        import threading
        from types import SimpleNamespace

        self._lock = threading.Lock()
        self._SimpleNamespace = SimpleNamespace
        self.points: dict[str, tuple[list, dict]] = {}
        self.fail_ids = set(fail_ids)
        self.calls = 0
        self.jitter = jitter

    def collection_exists(self, collection_name):
        return False

    def create_collection(self, collection_name, vectors_config, quantization_config=None):
        self.quantization_config = quantization_config
        return True

    def upsert(self, collection_name, points, wait):
        import time as time_module

        with self._lock:
            self.calls += 1
            call = self.calls
        if self.jitter:
            time_module.sleep(0.01 * (call % 4))
        for point in points:
            if point.payload.get("item_id") in self.fail_ids:
                raise RuntimeError(f"simulated upsert failure for {point.payload['item_id']}")
        with self._lock:
            for point in points:
                self.points[point.id] = (list(point.vector), dict(point.payload))
        return True

    def count(self, collection_name, exact):
        with self._lock:
            return self._SimpleNamespace(count=len(self.points))

    def scroll(self, collection_name, limit, with_payload, with_vectors):
        with self._lock:
            point_id = next(iter(self.points))
            vector, payload = self.points[point_id]
        return [self._SimpleNamespace(id=point_id, payload=payload, vector=vector)], None

    def query_points(self, collection_name, query, limit, with_payload):
        with self._lock:
            point_id = next(iter(self.points))
            payload = self.points[point_id][1]
        return self._SimpleNamespace(points=[self._SimpleNamespace(payload=payload)])

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
