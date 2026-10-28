import unittest
from pathlib import Path

from vibiomir.config import load_config
from vibiomir.query import build_query_processor
from vibiomir.reranking import build_reranker
from vibiomir.retrieval import build_retriever


class FullConfigTest(unittest.TestCase):
    def test_every_full_pipeline_has_constructible_components(self):
        paths = sorted(Path("configs/full").glob("*.yaml"))
        self.assertEqual(len(paths), 10)
        for path in paths:
            with self.subTest(config=path.name):
                config = load_config(path)
                build_query_processor(config["query"])
                for retriever in config.get("document_retrievers", []):
                    build_retriever(retriever)
                for retriever in config.get("chunk_retrievers", []):
                    build_retriever(retriever)
                build_reranker(config["reranker"])
                self.assertFalse(config["retrieval"]["cascade"])


if __name__ == "__main__":
    unittest.main()
