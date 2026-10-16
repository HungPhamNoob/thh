import unittest

from vibiomir.io.submission import validate_submission
from vibiomir.schemas import Document


class SubmissionTest(unittest.TestCase):
    def test_generated_chunk_is_rejected(self):
        rows = [
            {
                "id": 1,
                "relevant_docs": ["d1"],
                "relevant_chunks": [{"doc_id": "d1", "chunk_text": "a paraphrase"}],
            }
        ]
        errors = validate_submission(rows, [Document(id="d1", text="exact source text")])
        self.assertTrue(any("exact source substring" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

