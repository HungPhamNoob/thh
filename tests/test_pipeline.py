import unittest
from pathlib import Path

from vibiomir.config import load_config
from vibiomir.io.dataset import load_documents, load_queries
from vibiomir.pipeline import SearchPipeline


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_bm25_pipeline_runs_end_to_end(self):
        config = load_config(ROOT / "configs/experiments/00_bm25_dual_level.yaml")
        config["chunking"]["min_tokens"] = 1
        pipeline = SearchPipeline(config)
        pipeline.fit(load_documents(ROOT / "examples/corpus.jsonl"))
        predictions = pipeline.run(load_queries(ROOT / "examples/queries.jsonl"))
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(prediction.relevant_docs for prediction in predictions))
        for prediction in predictions:
            for chunk in prediction.relevant_chunks:
                self.assertIn(chunk.text, pipeline.documents[chunk.doc_id].text)


if __name__ == "__main__":
    unittest.main()

