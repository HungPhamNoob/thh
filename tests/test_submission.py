import unittest

from vibiomir.io.submission import validate_submission
from vibiomir.schemas import Chunk, Document, Prediction


class SubmissionTest(unittest.TestCase):
    def test_writer_uses_numeric_official_ids(self):
        prediction = Prediction(
            query_id="1",
            relevant_docs=["101"],
            relevant_chunks=[
                Chunk(
                    id="101:0",
                    doc_id="101",
                    text="source text",
                    index_text="source text",
                    order=0,
                    start_char=0,
                    end_char=11,
                )
            ],
        )
        row = prediction.submission_dict()
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["relevant_docs"], [101])
        self.assertEqual(row["relevant_chunks"][0]["doc_id"], 101)

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
