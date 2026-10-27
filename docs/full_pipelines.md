# Chạy pipeline full-corpus và so sánh theo metric BTC

## 1. Kết luận về nơi lưu chunk

BTC chỉ phát `query.parquet` và `links_corpus_vi_en_zh.parquet`; thể lệ không yêu cầu Qdrant
Cloud. Project dùng local artifacts làm mặc định:

```text
data/processed/corpus/                 parquet gốc sau ingest
artifacts/full/<method>/store.sqlite   document, chunk gốc, offset, language
artifacts/full/<method>/*_index/       Lucene / FAISS
artifacts/full/<method>/colbert/       PLAID index
runs/<method>/submission.json          output đúng schema BTC
runs/<method>/metrics.json             doc P/R/F2, chunk P/R/F2, final_score
```

SQLite là nguồn duy nhất của `chunk_text` nộp bài. Lucene/FAISS/PLAID chỉ giữ representation
để tìm kiếm và có thể build lại. Vì vậy không cần Qdrant; dùng Qdrant Cloud còn thêm chi phí,
network latency và rủi ro lệch mapping id mà không đem lại lợi ích cho batch benchmark này.

## 2. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ingest,dense,transformers,bge,lucene,colbert]'
# Có thể cài ColBERT trực tiếp từ repo chính chủ để pin đúng source revision.
# Pyserini >= 0.44 cần Java 21.
```

Model nặng được import lười. BM25 cần extra `lucene`; dense cần `dense`; MedCPT,
BMRetriever, Qwen3, SPLADE và Late Chunking cần `transformers`; BGE-M3 cần `bge`.

## 3. Ingest

Mỗi bước dài có state/part riêng và có thể chạy lại:

```bash
bash scripts/ingest_all.sh prepare
bash scripts/ingest_all.sh pubmed
bash scripts/ingest_all.sh crawl
bash scripts/ingest_all.sh wayback
bash scripts/ingest_all.sh merge
```

Corpus cuối là `data/processed/corpus/`. Không chạy song song `merge` với job đang ghi part.

## 4. Build từng phương pháp

Tên method là tên file không có `.yaml` trong `configs/full/`:

```bash
bash scripts/build_method.sh 00_bm25_lucene data/processed/corpus
bash scripts/build_method.sh 01_mgte_dense data/processed/corpus
bash scripts/build_method.sh 02_bge_m3 data/processed/corpus
# ... tương tự đến 08_late_chunking_en
```

Input có thể lấy trực tiếp từ một file parquet trong Hugging Face Dataset, có pin revision:

```bash
bash scripts/build_method_from_hf.sh 01_mgte_dense \
  owner/dataset corpus.parquet <dataset-commit-sha>
```

Hoặc gọi CLI rõ từng bước:

```bash
vibiomir index prepare --config configs/full/01_mgte_dense.yaml \
  --corpus data/processed/corpus \
  --store artifacts/full/01_mgte_dense/store.sqlite
python scripts/build_method_indexes.py \
  --config configs/full/01_mgte_dense.yaml \
  --store artifacts/full/01_mgte_dense/store.sqlite
```

Các config full đều chạy chunk-first để index tìm trên toàn corpus. Post-filter một FAISS top-k
theo vài trăm document của cascade không bảo đảm recall; vì vậy không dùng cách đó cho so sánh
chuẩn. Document score được aggregate từ chunk hits rồi chấm đồng thời với chunk score.

## 5. Retrieve, validate và metric

```bash
bash scripts/run_method.sh 01_mgte_dense \
  data/raw/official/query.parquet path/to/dev_gold.json
```

Khi có `gold`, lệnh ghi:

- macro document precision/recall/F2 theo từng query;
- macro chunk precision/recall/F2 với cả hai điều kiện LCS ≥ 0,4;
- `final_score = (document_f2 + chunk_f2) / 2`;
- `submission.json` được validate schema. Trước khi nộp thật, chạy thêm validate với corpus để
  kiểm tra `chunk_text` là substring nguyên văn.

## 6. Smoke test

```bash
bash scripts/smoke_all.sh
```

Smoke suite chạy unit test, parse/factory-check cả 9 config, build SQLite + persistent exact
matrix index trên `examples/`, retrieve, evaluate và validate end-to-end. Nó cố ý dùng hash
encoder nhỏ, không tải checkpoint nhiều GB. Việc tải model và encode 1,2M trang là production
job GPU, không phải smoke test CI.

## 7. Chi phí và so sánh công bằng

- Giữ cùng chunker, selector và dev queries khi so first-stage methods.
- Ghi commit code, config resolve, model revision, corpus version và metrics vào mỗi run.
- MedCPT/BMRetriever/ColBERTv2/SPLADE/Late Chunking chỉ đánh English slice sau NLLB; không so
  điểm tổng với model đa ngữ mà không ghi rõ khác biệt coverage.
- BGE-M3 và Qwen3 có reranker riêng; muốn đo first-stage thuần thì override
  `--set reranker.type=none` và giữ cùng candidate/output k.
