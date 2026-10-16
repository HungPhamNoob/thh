import unittest

from vibiomir.fusion import ReciprocalRankFusion
from vibiomir.schemas import SearchResult


class FusionTest(unittest.TestCase):
    def test_weighted_rrf(self):
        runs = {
            "lexical": [SearchResult("a", 8.0, rank=1), SearchResult("b", 7.0, rank=2)],
            "dense": [SearchResult("b", 0.9, rank=1), SearchResult("c", 0.8, rank=2)],
        }
        results = ReciprocalRankFusion(rank_constant=10).fuse(
            runs, weights={"lexical": 1.0, "dense": 2.0}
        )
        self.assertEqual(results[0].item_id, "b")
        self.assertIn("fusion", results[0].payload)


if __name__ == "__main__":
    unittest.main()

