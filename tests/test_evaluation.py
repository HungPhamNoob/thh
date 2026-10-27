import unittest

from vibiomir.evaluation import evaluate_submission, f_beta, lcs_overlap, lcs_overlaps


class EvaluationTest(unittest.TestCase):
    def test_official_normalization_and_overlap(self):
        self.assertEqual(lcs_overlap("A  &amp;  B C", "a & b c"), 1.0)

    def test_f2_prefers_recall(self):
        self.assertGreater(f_beta(0.5, 1.0), f_beta(1.0, 0.5))

    def test_chunk_precision_does_not_use_one_to_one_matching(self):
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
        self.assertEqual(metrics["chunk_precision"], 1.0)
        self.assertEqual(metrics["chunk_recall"], 1.0)

    def test_both_overlap_denominators_must_pass(self):
        gold = "alpha beta gamma delta"
        candidate = "noise words before alpha beta gamma delta and many words after it"
        overlap_gold, overlap_pred = lcs_overlaps(candidate, gold)
        self.assertEqual(overlap_gold, 1.0)
        self.assertLess(overlap_pred, 0.4)

    def test_f2_is_macro_averaged_per_query(self):
        gold = [
            {"id": 1, "relevant_docs": ["a", "c"], "relevant_chunks": []},
            {"id": 2, "relevant_docs": ["b"], "relevant_chunks": []},
        ]
        predictions = [
            {"id": 1, "relevant_docs": ["a"], "relevant_chunks": []},
            {"id": 2, "relevant_docs": ["b", "noise"], "relevant_chunks": []},
        ]
        metrics = evaluate_submission(predictions, gold)
        expected = (f_beta(1.0, 0.5) + f_beta(0.5, 1.0)) / 2.0
        self.assertAlmostEqual(metrics["document_f2"], expected)
        self.assertNotEqual(
            metrics["document_f2"],
            f_beta(metrics["document_precision"], metrics["document_recall"]),
        )


if __name__ == "__main__":
    unittest.main()
