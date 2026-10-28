# Dùng `vibiomir-corpus/vibiomir-corpus` và tạo bài nộp BTC

## Mốc dữ liệu cố định

Production runner pin corpus ở commit:

```text
dataset:  vibiomir-corpus/vibiomir-corpus
revision: 13d60924d53e720a7f84c78e3b08b7e17b670b5d
pattern:  corpus/v1/*.parquet
shards:   13
rows:     1,220,598
```

Nguồn kiểm tra: [dataset tree](https://huggingface.co/datasets/vibiomir-corpus/vibiomir-corpus/tree/13d60924d53e720a7f84c78e3b08b7e17b670b5d),
[`corpus_info.json`](https://huggingface.co/datasets/vibiomir-corpus/vibiomir-corpus/blob/13d60924d53e720a7f84c78e3b08b7e17b670b5d/corpus_info.json)
và [`coverage.csv`](https://huggingface.co/datasets/vibiomir-corpus/vibiomir-corpus/blob/13d60924d53e720a7f84c78e3b08b7e17b670b5d/coverage.csv).

Dataset đủ 1.220.598 ID nhưng chỉ 1.047.415 dòng có text. 173.183 dòng text rỗng được bỏ khỏi
SQLite/Qdrant vì không có nội dung để chunk hoặc embed. Runner kiểm tra đúng các số này và dừng
ngay nếu dataset revision trả về số shard/row khác dự kiến.

## Luồng chạy

```text
13 Parquet shards (HF, pinned SHA)
  -> verify 1,220,598 rows
  -> bỏ 173,183 text rỗng
  -> 1,047,415 documents có text
  -> chunk nguyên văn + store.sqlite
  -> embedding -> Qdrant / Lucene / PLAID
  -> retrieve 1,200 queries
  -> submission.json
  -> strict validation
  -> submission.zip
```

Không dùng LLM để điền 173.183 tài liệu thiếu. Nội dung sinh hoặc diễn giải lại không phải source
text và không được phép xuất hiện trong `chunk_text`. Phần thiếu phải được crawl bổ sung rồi merge
thành corpus revision mới.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[ingest,dense,transformers,bge,dev]'

cp .env.example .env
# điền QDRANT_URL và QDRANT_API_KEY trong .env
```

Không commit `.env` hoặc API key. Hugging Face snapshot được cache nên các method sau không tải lại
1,43 GB nếu vẫn dùng cùng revision.

## Build một method

Ví dụ mGTE multilingual:

```bash
bash scripts/build_vibiomir_method.sh 01_mgte_dense
```

Script thực hiện hai bước:

1. Tải/đọc đủ 13 shard, chunk và tạo `artifacts/full/01_mgte_dense/store.sqlite`.
2. Encode chunk và upsert collection `mgte_dense_chunks` lên Qdrant.

BM25 không cần Qdrant:

```bash
bash scripts/build_vibiomir_method.sh 00_bm25_lucene
```

## Sinh bài nộp

```bash
bash scripts/run_method.sh \
  01_mgte_dense \
  data/raw/official/query.parquet
```

Nếu có gold dev:

```bash
bash scripts/run_method.sh \
  01_mgte_dense \
  data/raw/official/query.parquet \
  data/raw/official/dev_gold.json
```

Output:

```text
runs/01_mgte_dense/submission.json
runs/01_mgte_dense/submission.zip   <- file tải lên hệ thống BTC
runs/01_mgte_dense/metrics.json     <- chỉ có ý nghĩa khi cung cấp gold
```

Validator đảm bảo:

- output có đúng toàn bộ ID trong query parquet (bộ chính thức là 1.200 query);
- query ID, document ID và chunk document ID là integer;
- mỗi row luôn có `id`, `relevant_docs`, `relevant_chunks`;
- `chunk_order` nếu có phải là integer không âm;
- `chunk_text` là substring của document tương ứng trong SQLite;
- ZIP chỉ chứa duy nhất `submission.json`, không có thư mục con.

Đóng gói lại một JSON đã có:

```bash
vibiomir package \
  --submission runs/01_mgte_dense/submission.json \
  --queries data/raw/official/query.parquet \
  --store artifacts/full/01_mgte_dense/store.sqlite \
  --output runs/01_mgte_dense/submission.zip
```

## Corpus chưa hoàn chỉnh

Không nên crawl lại toàn bộ 1,22 triệu URL. Ưu tiên bổ sung các domain thiếu nhiều nhất, merge bằng
ID chính thức, phát hành corpus `v2`, rồi rebuild/upsert collection. `120ask.com` là điểm nghẽn lớn
nhất. Cho đến khi có v2, mọi metric phải ghi rõ đang dùng corpus v1 coverage 85,81% có text.

Dataset hiện đang public dù README ghi internal/private. Nếu quyền phân phối text chưa rõ, chủ
dataset cần chuyển repository sang private/gated trước khi tiếp tục chia sẻ.
