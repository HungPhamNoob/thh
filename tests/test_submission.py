import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vibiomir.io.submission import (
    validate_submission,
    validate_submission_zip,
    write_submission_zip,
)
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

    def test_official_validation_requires_all_queries_and_numeric_ids(self):
        rows = [{"id": "1", "relevant_docs": [], "relevant_chunks": []}]
        errors = validate_submission(
            rows,
            expected_query_ids={"1", "2"},
            require_numeric_ids=True,
            strict_fields=True,
        )
        self.assertTrue(any("expected integer" in error for error in errors))
        self.assertTrue(any("missing query ids" in error for error in errors))

    def test_submission_zip_has_one_root_level_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission = root / "predictions.json"
            submission.write_text(json.dumps([]), encoding="utf-8")
            output = root / "submission.zip"
            write_submission_zip(submission, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                self.assertEqual(json.loads(archive.read("submission.json")), [])

    def test_strict_validator_rejects_all_cross_reference_errors(self):
        rows = [
            {
                "id": 1,
                "relevant_docs": [101, 101, 999],
                "relevant_chunks": [
                    {"doc_id": 102, "chunk_text": "source", "chunk_order": -1, "extra": 1}
                ],
                "extra": True,
            },
            {"id": 1, "relevant_docs": [], "relevant_chunks": []},
            {"id": 3, "relevant_docs": [], "relevant_chunks": []},
        ]
        errors = validate_submission(
            rows,
            [Document(id="101", text="source")],
            expected_query_ids={"1", "2"},
            require_numeric_ids=True,
            strict_fields=True,
        )
        joined = "\n".join(errors)
        self.assertIn("duplicate query id", joined)
        self.assertIn("duplicate ids", joined)
        self.assertIn("unknown doc_id 999", joined)
        self.assertIn("absent from relevant_docs", joined)
        self.assertIn("non-negative integer", joined)
        self.assertIn("unexpected fields", joined)
        self.assertIn("missing query ids", joined)
        self.assertIn("unexpected query ids", joined)

    def test_zip_validator_rejects_multiple_and_nested_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, members in (
                ("multiple.zip", {"submission.json": "[]", "README": "x"}),
                ("nested.zip", {"folder/submission.json": "[]"}),
            ):
                output = root / name
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for member, content in members.items():
                        archive.writestr(member, content)
                self.assertTrue(validate_submission_zip(output))


if __name__ == "__main__":
    unittest.main()
