import unittest

from vibiomir.evaluation import evaluate_submission, f_beta, lcs_overlap


class EvaluationTest(unittest.TestCase):
    def test_official_normalization_and_overlap(self):
        self.assertEqual(lcs_overlap("A  &amp;  B C", "a & b c"), 1.0)

    def test_f2_prefers_recall(self):
        self.assertGreater(f_beta(0.5, 1.0), f_beta(1.0, 0.5))

    def test_one_to_one_chunk_matching(self):
        gold = [
            {
                "id": 1,
                "relevant_docs": ["d"],
                "relevant_chunks": [{"doc_id": "d", "chunk_text": "alpha beta gamma"}],
            }
        ]
        predictions = [
            {
                "id": 1,
                "relevant_docs": ["d"],
                "relevant_chunks": [
                    {"doc_id": "d", "chunk_text": "alpha beta gamma"},
                    {"doc_id": "d", "chunk_text": "alpha beta gamma"},
                ],
            }
        ]
        metrics = evaluate_submission(predictions, gold)
        self.assertEqual(metrics["chunk_precision"], 0.5)
        self.assertEqual(metrics["chunk_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

