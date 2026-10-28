# Kiểm chứng nguồn và phạm vi triển khai

Tài liệu này thay cho bảng ánh xạ cũ. Một tên paper chỉ được gắn vào code khi công thức
inference và checkpoint/index implementation đã đối chiếu với nguồn của tác giả. Các config
full-corpus nằm trong `configs/full/`; cách chạy ở [`full_pipelines.md`](full_pipelines.md).

## Phương pháp đã triển khai từ nguồn chính chủ

| Phương pháp | Nguồn code/model đã đối chiếu | Phần triển khai trung thực | Config |
|---|---|---|---|
| BM25/Lucene | `castorini/pyserini` | `JsonCollection`, Lucene BM25, index cục bộ | `00_bm25_lucene` |
| mGTE | model card `Alibaba-NLP/gte-multilingual-base` và reranker | dense đa ngữ, normalize embedding, HF sequence-classification reranker | `01_mgte_dense` |
| BGE-M3 | `FlagOpen/FlagEmbedding`, `BAAI/bge-m3` | dense Qdrant + lexical impact; `compute_score` dense/sparse/ColBERT chính chủ trên candidate pool | `02_bge_m3` |
| MedCPT | model card NCBI của query/article/cross encoders | hai tower riêng, CLS, query 64 token, article pair 512 token, raw dot product; chỉ index tiếng Anh | `03_medcpt_en` |
| BMRetriever | `ritaranx/BMRetriever` | task instruction, passage prefix, thêm EOS, last-token pooling, raw dot product; chỉ tiếng Anh | `04_bmretriever_en` |
| Qwen3 Embedding/Reranker | `QwenLM/Qwen3-Embedding` | instruction query, document không instruction, left padding + EOS + last token + L2; reranker causal-LM xác suất yes/no | `05_qwen3` |
| ColBERTv2 + PLAID | `stanford-futuredata/ColBERT` | `Indexer`/`Searcher` và compressed PLAID index của tác giả, map PID về chunk id | `06_colbertv2_en` |
| SPLADE++ | `naver/splade` | `max(log(1+ReLU(logits)))`, lượng tử ×100 như script tác giả, `JsonVectorCollection --impact --pretokenized` | `07_splade_v2_en` |
| Late Chunking | `jina-ai/late-chunking` | encode document dài trước, pool token theo span chunk; hỗ trợ cửa sổ overlap của repo khi quá context | `08_late_chunking_en` |

Các nhánh tiếng Anh dùng NLLB để tạo `translate_en` rồi chỉ đưa view đó vào retriever. Đây là
bước thích nghi dữ liệu của pipeline, không được ghi là đóng góp của MedCPT/BMRetriever/
ColBERT/SPLADE/Late Chunking.

## Những gì không gắn nhãn thành implementation

| Nguồn trong Drive | Quyết định | Lý do |
|---|---|---|
| RAPTOR/Dynamic RAPTOR | không triển khai | `RecursiveChunker` chỉ là chia text theo separator; RAPTOR cần embedding, clustering, cây và summary phân cấp. Hai thứ không tương đương. |
| PARADE | không triển khai | `DocumentAggregator` hiện là max/top-k mean/logsumexp, không phải transformer passage aggregation được huấn luyện của PARADE. |
| Query2doc, GRF, BRIGHT | không đưa vào bộ full | prompt và model sinh giả trong config cũ không tái tạo implementation/teacher của paper; giữ config cũ chỉ như ablation nội bộ, không tuyên bố reproduce paper. |
| ConTEB/InSeNT | không triển khai | thêm cửa sổ lân cận vào `index_text` không đủ để gọi là InSeNT. |
| ColBERT-XM, Jina-ColBERT-v2, XTR | chưa đưa vào full | chưa xác minh được một đường index/search tương thích checkpoint và corpus 1,2M trong code hiện tại. Không thay bằng BGE `mode: colbert`. |
| SapBERT, BioLORD | không coi là retriever hoàn chỉnh | đây là representation/concept supervision; cần ontology/entity linker và recipe đánh giá riêng. |
| GPL, SWIM-IR, DR.DECR, InPars | chỉ là training recipe | cần sinh dữ liệu, teacher, hard negatives và huấn luyện; không thể thay bằng một query expander lúc inference. |
| MiLQ, ViRE, Vi–En CLIR, MIRACL, BEIR | chỉ dùng để thiết kế/evaluate | benchmark hoặc nghiên cứu phân tích, không phải một component inference có thể gắn nhãn trực tiếp. |
| Revela | chưa triển khai | cần pipeline self-supervised training riêng. |
| CeQe 2608, BioASQ 2608/2609, Clinical-reranking 2610, SAFE-QE 2610 | loại | phát hành sau hạn model/paper 01/08/2026 của cuộc thi. |

## Ranh giới tính đúng đắn

- BGE-M3 multi-vector được dùng làm exact reranker trên candidate pool; không giả vờ rằng
  Qdrant dense là một PLAID index.
- SPLADE/BGE lexical dùng impact index; BM25 dùng index riêng. Không đưa sparse weight vào
  `DefaultLuceneDocumentGenerator` của BM25.
- MedCPT và BMRetriever không chạy trực tiếp trên tài liệu Việt/Trung.
- ColBERTv2 dùng checkpoint tiếng Anh; config không gọi nó là multilingual ColBERT-XM.
- `chunk_text` luôn lấy từ SQLite store và là substring của document gốc. Model chỉ score
  `index_text`; không có text sinh bởi model đi vào submission.

