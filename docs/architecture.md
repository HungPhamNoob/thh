# Kiến trúc và contract

## Các object chính

| Object | Trường quan trọng | Ý nghĩa |
|---|---|---|
| `Document` | `id`, `text`, `language`, `url` | trang sau crawl |
| `Chunk` | `text`, `index_text`, `start_char`, `end_char` | đoạn gốc và representation dùng index |
| `QueryVariant` | `text`, `language`, `weight`, `variant_id` | original/translated/expanded query |
| `SearchItem` | `id`, `text`, `doc_id`, `payload` | unit chung cho retriever |
| `SearchResult` | `item_id`, `score`, `rank`, `source` | output chuẩn của mọi retriever |
| `Prediction` | documents, chunks | object duy nhất được writer chuyển thành submission |

## Vì sao tách module như vậy

- **Chunking** thay đổi cả recall của chunk và chất lượng embedding, nên là một trục ablation riêng.
- **Query processing** cần sinh nhiều view nhưng phải giữ weight riêng khi fusion.
- **Document và chunk retrieval** dùng index khác nhau. Cascade giảm chi phí rerank trên 1.2M trang.
- **Fusion** xử lý scale score khác nhau; reranker chỉ nhìn một candidate pool đã kiểm soát.
- **Aggregation** cần thiết vì metric chấm đồng thời document và chunk. MaxP, top-k mean và log-sum-exp là ba giả thuyết khác nhau.
- **Selection** ảnh hưởng trực tiếp F2. `AdaptiveGapSelector` cho số lượng kết quả khác nhau theo độ chắc chắn của query.
- **Evaluation** nằm độc lập để một thay đổi pipeline không thay đổi metric.

## Hai chế độ retrieval

### Cascade

1. Query variants chạy qua document retrievers.
2. Fusion lấy candidate documents.
3. Chunk retrievers chỉ score chunks thuộc candidate documents.
4. Reranker xử lý candidate chunks.
5. Passage scores được aggregate về document scores.
6. Fusion document-stage và chunk-stage.

### Chunk-first

Đặt `retrieval.cascade: false`. Chunk retrievers search toàn corpus; document score được suy ra từ chunks. Cách này tăng recall nhưng tốn index/search hơn.

## Invariant submission

`chunk_text == document.text[start_char:end_char]` đối với mọi chunk do các chunker lõi tạo. Contextual chunking chỉ sửa `index_text`. Lệnh `validate --corpus ...` kiểm tra lại invariant bằng substring trước khi nộp.

## Điểm cần scale khi dùng full corpus

- In-memory BM25 chỉ dành cho smoke test và sample.
- Full BM25 dùng `JsonCollection`; learned sparse dùng `JsonVectorCollection --impact --pretokenized`. Hai index không dùng chung generator.
- Dense index dùng FAISS; HNSW dễ bắt đầu, IVF/PQ phù hợp khi RAM hạn chế. Vector và id map nằm cạnh manifest của index.
- BGE-M3 full-corpus dùng dense FAISS + lexical impact để sinh candidate, sau đó gọi `compute_score` chính chủ cho dense/sparse/ColBERT trên candidate pool.
- ColBERTv2 dùng `Indexer`/`Searcher` PLAID của repo tác giả; không gọi BGE multi-vector là ColBERTv2.
- Local `ArtifactStore` (SQLite) giữ exact chunk text và offset. BTC không yêu cầu Qdrant; xem [`full_pipelines.md`](full_pipelines.md).
- Cache chunks và embeddings theo hash của config để không encode lại giữa ablation.

