import unittest
from pathlib import Path

from vibiomir.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_component_replace_removes_incompatible_parent_keys(self):
        config = load_config(ROOT / "configs/experiments/07_recursive_chunking.yaml")
        self.assertEqual(config["chunking"]["type"], "recursive")
        self.assertNotIn("overlap_sentences", config["chunking"])


if __name__ == "__main__":
    unittest.main()

