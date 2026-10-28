import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from vibiomir.io.submission import (
    load_submission,
    referenced_doc_ids,
    validate_submission_streaming,
    write_submission,
    write_submission_zip,
)
from vibiomir.schemas import Chunk, Prediction


def _corpus_jsonl(path: Path) -> None:
    rows = [
        {"id": "d1", "text": "exact source text about dengue fever"},
        {"id": "d2", "text": "another document"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class StreamingValidationTest(unittest.TestCase):
    def test_accepts_exact_substring_and_rejects_paraphrase(self):
        with TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            _corpus_jsonl(corpus)
            good = [{"id": 1, "relevant_docs": [1], "relevant_chunks": [
                {"doc_id": "d1", "chunk_text": "source text about dengue"}]}]
            self.assertEqual(validate_submission_streaming(good, corpus), [])
            bad = [{"id": 1, "relevant_docs": [1], "relevant_chunks": [
                {"doc_id": "d1", "chunk_text": "a paraphrase"},
                {"doc_id": "missing", "chunk_text": "text"}]}]
            errors = validate_submission_streaming(bad, corpus)
            self.assertTrue(any("exact source substring" in error for error in errors))
            self.assertTrue(any("unknown doc_id missing" in error for error in errors))

    def test_referenced_doc_ids(self):
        rows = [{"id": 1, "relevant_chunks": [{"doc_id": "d1"}, {"doc_id": 7}]}]
        self.assertEqual(referenced_doc_ids(rows), {"d1", "7"})


class PackageZipTest(unittest.TestCase):
    def test_zip_has_single_root_json(self):
        with TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            prediction = Prediction(
                query_id="1",
                relevant_docs=["d1"],
                relevant_chunks=[
                    Chunk(
                        id="d1:0",
                        doc_id="d1",
                        text="exact source text",
                        index_text="exact source text",
                        order=0,
                        start_char=0,
                        end_char=17,
                    )
                ],
            )
            write_submission([prediction], submission)
            zip_path = write_submission_zip(submission, Path(tmp) / "out" / "submission.zip")
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                self.assertEqual(
                    load_submission(submission),
                    json.loads(archive.read("submission.json").decode("utf-8")),
                )
            self.assertTrue(zip_path.exists())


if __name__ == "__main__":
    unittest.main()
