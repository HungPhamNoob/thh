import unittest

from vibiomir.chunking import ContextualWindowChunker, FixedTokenChunker, SentenceChunker
from vibiomir.schemas import Document


class ChunkingTest(unittest.TestCase):
    def test_fixed_chunks_are_exact_substrings(self):
        document = Document(id="d1", text="Một hai ba bốn năm sáu bảy tám chín mười.")
        chunks = FixedTokenChunker(size=4, overlap=1, min_tokens=1).split(document)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.text, document.text[chunk.start_char : chunk.end_char])

    def test_context_is_index_only(self):
        document = Document(
            id="d1",
            title="Bệnh học",
            text="Câu thứ nhất có bối cảnh. Câu thứ hai là bằng chứng cần tìm.",
        )
        chunker = ContextualWindowChunker(
            base=SentenceChunker(max_tokens=6, overlap_sentences=0, min_tokens=1),
            surrounding_chars=30,
        )
        chunks = chunker.split(document)
        self.assertTrue(any(chunk.index_text != chunk.text for chunk in chunks))
        for chunk in chunks:
            self.assertIn(chunk.text, document.text)


if __name__ == "__main__":
    unittest.main()

