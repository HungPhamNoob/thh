# ViBioMIR Lab

Codebase thử nghiệm retrieval cho R2AI2026/ViBioMIR: 1.2M trang y sinh bằng tiếng Việt, Anh và Trung; đầu ra gồm cả `doc_id` và đoạn văn gốc.

Mục tiêu của repo là thay từng khối bằng YAML để chạy ablation. Baseline nhẹ chạy chỉ với `numpy` và `PyYAML`; model GPU được nạp lười qua optional dependencies.

## Chạy nhanh

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

vibiomir run \
  --config configs/experiments/00_bm25_dual_level.yaml \
  --corpus examples/corpus.jsonl \
  --queries examples/queries.jsonl \
  --output runs/demo/submission.json \
  --set chunking.min_tokens=1

vibiomir validate \
  --submission runs/demo/submission.json \
  --corpus examples/corpus.jsonl

vibiomir evaluate \
  --predictions runs/demo/submission.json \
  --gold examples/gold.json
```

Chạy test không cần cài `pytest`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Luồng dữ liệu

```mermaid
flowchart TD
    A["Document gốc"] --> B["Chunker"]
    B --> C["text: đoạn gốc"]
    B --> D["index_text: thêm context"]
    Q["Query Việt"] --> E["Dịch / mở rộng / reasoning"]
    E --> F["Document retrievers"]
    E --> G["Chunk retrievers"]
    F --> H["Candidate documents"]
    H --> G
    G --> I["Fusion + rerank"]
    I --> J["Passage → document aggregation"]
    J --> K["F2 cutoff"]
    C --> L["Submission validator"]
    K --> L
```

`Chunk.text` luôn là substring của trang đã crawl. `Chunk.index_text` có thể chứa title, context hoặc representation khác để retrieval. Submission chỉ ghi `Chunk.text`, nhờ vậy không vô tình nộp summary hoặc đoạn do LLM tạo.

## Cấu trúc

```text
src/vibiomir/
├── chunking/       fixed, sentence, recursive, contextual
├── query/          identity, synonym, NLLB, local-LLM expansion
├── indexing/       SQLite artifact store, FAISS/impact/PLAID export
├── retrieval/      BM25, FAISS, BGE-M3, SPLADE, ColBERTv2, Late Chunking
├── fusion/         weighted RRF, normalized CombSUM
├── reranking/      identity, HF classifier, BGE-M3 multi-vector, Qwen3
├── aggregation/    MaxP, top-k mean, log-sum-exp
├── selection/      threshold, adaptive score gap, lexical MMR
├── pipeline/       document→chunk cascade và chunk-first
├── evaluation/     normalization, LCS overlap, macro F2
└── io/             JSONL/parquet/Hugging Face input, writer và validator
```

Chi tiết interface: [`docs/architecture.md`](docs/architecture.md). Bảng kiểm chứng paper/source: [`docs/source_map.md`](docs/source_map.md). Hướng dẫn full-corpus: [`docs/full_pipelines.md`](docs/full_pipelines.md).

## Config và override

Config experiment kế thừa `configs/base.yaml`. Có thể đổi tham số mà không tạo file mới:

```bash
vibiomir show-config \
  --config configs/experiments/08_f2_adaptive_cutoff.yaml \
  --set chunking.max_tokens=256 \
  --set retrieval.chunk_pool=1000
```

Các config trong `configs/experiments/` là ablation in-memory trên subset. Chúng không được
coi là reproduction paper. Chín pipeline full-corpus, đã tách index và source audit, nằm trong
`configs/full/`:

| Config full | Phương pháp |
|---|---|
| `00_bm25_lucene` | BM25/Lucene |
| `01_mgte_dense` | mGTE + multilingual reranker |
| `02_bge_m3` | dense + learned sparse + exact three-view candidate rerank |
| `03_medcpt_en` | NLLB → English MedCPT |
| `04_bmretriever_en` | NLLB → English BMRetriever |
| `05_qwen3` | Qwen3 embedding + reranker |
| `06_colbertv2_en` | NLLB → ColBERTv2/PLAID |
| `07_splade_v2_en` | NLLB → SPLADE++ impact index |
| `08_late_chunking_en` | NLLB → Jina Late Chunking |

Kế hoạch chạy baseline theo pha và ablation: [`docs/experiment_guide.md`](docs/experiment_guide.md).

## Dữ liệu đầu vào

Corpus có thể là JSONL, một file/thư mục parquet, hoặc một file parquet tải trực tiếp từ Hugging
Face Dataset. Ví dụ JSONL:

```json
{"id":"583","url":"https://...","title":"...","language":"vi","text":"nội dung sạch nhưng giữ nguyên ký tự gốc"}
```

Query dùng JSONL:

```json
{"id":1,"language":"vi","text":"Câu hỏi y sinh..."}
```

Để build một pipeline đầy đủ trên 1,2M documents/chunks:

```bash
bash scripts/build_method.sh 00_bm25_lucene data/processed/corpus
bash scripts/run_method.sh 00_bm25_lucene \
  data/raw/official/query.parquet path/to/dev_gold.json
```

`store.sqlite` giữ chunk gốc; Lucene/FAISS/PLAID được lưu local dưới `artifacts/full/`. BTC
không yêu cầu Qdrant Cloud.

## Optional dependencies

```bash
pip install -e '.[dense]'        # SentenceTransformers, CrossEncoder, FAISS
pip install -e '.[bge]'          # BGE-M3 dense/sparse/ColBERT vectors
pip install -e '.[lucene]'       # Pyserini
pip install -e '.[colbert]'       # ColBERTv2/PLAID
pip install -e '.[transformers]' # NLLB và local LLM query expansion
```

Model tải về phải được kiểm tra lại theo luật contest: open weights, giới hạn tham số và ngày phát hành.

