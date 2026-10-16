# Kế hoạch thí nghiệm

## Pha 1: khóa evaluator và candidate budget

1. Chạy config 00.
2. Sweep `document_selection.max_k`, `chunk_selection.max_k`, `relative_threshold`.
3. Chọn theo `retrieval_mean_f2` để sweep nội bộ, đồng thời lưu document/chunk precision và recall riêng. Nếu BTC công bố công thức composite khác, dùng hai F2 thành phần để tính lại.
4. Kiểm tra query không có kết quả, duplicate URL và chunk provenance.

## Pha 2: retrieval family

Giữ chunking/cutoff cố định, thay từng family:

- BM25/Lucene.
- multilingual dense (mGTE).
- biomedical dense (BMRetriever, MedCPT sau translation).
- learned sparse (BGE-M3 sparse/SPLADE impact index).
- multi-vector (BGE-M3 ColBERT, ColBERT-XM/PyLate production index).

So sánh thêm recall tại candidate pool trước reranker. Nếu candidate recall thấp, reranker không thể sửa.

## Pha 3: query và chunk

Chạy one-factor-at-a-time:

- fixed 128/256/384 tokens, overlap 10–25%.
- sentence 128/180/256 tokens.
- recursive boundaries.
- contextual index text.
- original vs translation VI→EN/ZH.
- keywords, entities, reasoning, pseudo-document, và fusion của các views.

## Pha 4: rerank và aggregation

- candidate size 50/100/200/400.
- multilingual reranker vs MedCPT cross encoder.
- `max`, `topk_mean(k=2/3/5)`, `logsumexp`.
- document-stage only, chunk-stage only, fused stages.

## Báo cáo tối thiểu mỗi run

| Nhóm | Field |
|---|---|
| Reproducibility | config path, config đã resolve, git commit, seed |
| Cost | index time, search latency, peak RAM/VRAM |
| Candidate | Recall@50/100/250/500 nếu có qrels |
| Final | doc P/R/F2, chunk P/R/F2, retrieval mean F2 nội bộ |
| Slice | query language, retrieved document language, độ dài query |
| Error | no-hit, wrong document, right document/wrong chunk, boundary miss |

F2 ưu tiên recall nhưng không có nghĩa luôn tăng `k`: duplicate/near-duplicate chunks làm precision giảm nhanh. Tune document cutoff và số chunk mỗi document riêng.

