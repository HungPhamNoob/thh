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

    def test_export_jsonl_gzip_roundtrip(self):
        import gzip

        plain = self.root / "collection" / "part-00000.jsonl"
        compressed = self.root / "collection" / "part-00000.json.gz"
        with ArtifactStore(self.store_path) as store:
            expected = store.count("chunk")
            self.assertEqual(store.export_jsonl("chunk", plain), expected)
            self.assertEqual(store.export_jsonl("chunk", compressed), expected)
        with compressed.open("rb") as handle:
            self.assertEqual(handle.read(2), b"\x1f\x8b")  # gzip magic
        lines = gzip.decompress(compressed.read_bytes()).decode("utf-8").splitlines()
        self.assertEqual(lines, plain.read_text(encoding="utf-8").splitlines())
        self.assertIn("contents", json.loads(lines[0]))

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
