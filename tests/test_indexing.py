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


if __name__ == "__main__":
    unittest.main()
