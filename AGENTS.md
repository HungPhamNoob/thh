# AGENTS.md

Hướng dẫn chung cho mọi coding agent (Codex, Claude Code, Cursor, …) và người làm việc trên repo này. Đọc hết file này trước khi sửa code. Nếu cần thêm quy ước thì sửa file này, không tạo hướng dẫn riêng cho từng agent.

## 1. Dự án

`vibiomir-lab` là hệ thống truy hồi y sinh đa ngôn ngữ (Việt/Anh/Trung) cho cuộc thi R2AI2026 – ViBioMIR. Với 1.200 câu hỏi tiếng Việt, hệ thống trả về id tài liệu liên quan và đoạn trích nguyên văn trong 1.220.598 trang do BTC phát URL.

Đọc theo thứ tự:

1. [`docs/competition.md`](docs/competition.md): luật thi, metric chính xác, format nộp, mốc thời gian.
2. [`docs/plan.md`](docs/plan.md): kế hoạch, phân vai và backlog ticket (D*, E*, R*, P*). Mọi việc nên gắn với một ticket.
3. [`docs/data_ingestion.md`](docs/data_ingestion.md): thiết kế crawl/ingest corpus.
4. [`docs/architecture.md`](docs/architecture.md): kiến trúc pipeline retrieval và các object chính.
5. [`docs/experiment_guide.md`](docs/experiment_guide.md), [`docs/source_map.md`](docs/source_map.md), [`docs/drive_papers.md`](docs/drive_papers.md), [`docs/training_recipes.md`](docs/training_recipes.md) khi cần.

## 2. Cài đặt và lệnh thường dùng

Python ≥ 3.10. Core chỉ phụ thuộc `numpy` và `PyYAML`; model nặng nằm trong optional extras và được import lười.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .                     # thêm '.[dense]', '.[bge]', '.[lucene]', '.[transformers]', '.[dev]' khi cần

# test (không cần pytest)
PYTHONPATH=src python -m unittest discover -s tests -v
# PowerShell: $env:PYTHONPATH = "src"; python -m unittest discover -s tests -v

# smoke test end-to-end trên dữ liệu mẫu
vibiomir run --config configs/experiments/00_bm25_dual_level.yaml \
  --corpus examples/corpus.jsonl --queries examples/queries.jsonl \
  --output runs/demo/submission.json --set chunking.min_tokens=1
vibiomir validate --submission runs/demo/submission.json --corpus examples/corpus.jsonl
vibiomir evaluate --predictions runs/demo/submission.json --gold examples/gold.json
vibiomir show-config --config configs/experiments/08_f2_adaptive_cutoff.yaml --set chunking.max_tokens=256
```

Pyserini cần Java 21 (máy dev hiện có Java 17). Máy dev là laptop 16 GB RAM, không có GPU NVIDIA: việc encode embedding và rerank trên toàn corpus chạy trên GPU ngoài (Kaggle/Colab/máy thuê). Code phải chạy được smoke test bằng CPU trên `examples/`.

## 3. Bản đồ code

```text
src/vibiomir/
├── schemas.py      Document, Query, QueryVariant, Chunk, SearchItem, SearchResult, Prediction
├── text.py         official_normalize (chuẩn hoá theo luật chấm), lexical_tokens
├── config.py       YAML kế thừa `extends`, `_replace_: true`, override `--set a.b=c`
├── io/             đọc JSONL corpus/query, ghi và validate submission
├── chunking/       fixed, sentence, recursive, contextual (+ factory)
├── query/          identity, synonyms, NLLB, local LLM, composite (+ factory)
├── retrieval/      BM25 in-memory, SentenceTransformer/Qdrant, HF encoder, BGE-M3, Pyserini (+ factory)
├── fusion/         weighted RRF, normalized CombSUM
├── reranking/      identity, cross-encoder
├── aggregation/    chunk score → document score (max, topk_mean, logsumexp)
├── selection/      threshold_topk, adaptive_gap, mmr
├── pipeline/       SearchPipeline: cascade document → chunk
├── evaluation/     LCS overlap, P/R/F2
├── ingest/         tải file chính thức, manifest, crawler (SQLite state, rate limit), extract HTML,
│                   PubMed, Wayback, merge corpus
└── cli.py          vibiomir run | chunk | evaluate | validate | show-config
configs/            base.yaml, components/, experiments/NN_*.yaml
scripts/            build_pyserini_index.sh, run_sweep.py
```

Thêm một component mới (chunker, retriever, reranker, …):

1. Viết class kế thừa base class của module, dạng `@dataclass(slots=True)`, tham số khởi tạo trùng với key trong YAML.
2. Đăng ký `type` mới trong `factory.py` của module đó.
3. Import thư viện nặng bên trong `_load()` và báo `ImportError` chỉ rõ extra cần cài, theo mẫu các retriever có sẵn.
4. Thêm config trong `configs/components/` hoặc `configs/experiments/` và unit test nhỏ trong `tests/`.

`SearchPipeline.fit()` giữ toàn bộ document và chunk trong RAM, nên chỉ dùng cho sample. Chạy full corpus phải theo ticket R0 trong `docs/plan.md` (index offline + doc store).

## 4. Bất biến không được phá

1. **`chunk_text` phải là substring nguyên văn của text tài liệu** (`Chunk.text == document.text[start_char:end_char]`). `index_text` được phép thêm title/ngữ cảnh/bản dịch, nhưng submission chỉ ghi `Chunk.text`. Không bao giờ nộp text do model sinh hoặc viết lại.
2. **id tài liệu là `id` số nguyên trong `links_corpus`**. Submission phải có đủ 1.200 query, `id` là số nguyên, danh sách rỗng vẫn phải có. Chạy `vibiomir validate --corpus …` trước mọi lần nộp.
3. **Evaluator phải bám đúng công thức trong `docs/competition.md` §3.** Không sửa metric để số đẹp hơn. Nếu luật thay đổi, sửa doc trước rồi mới sửa code và test.
4. **Model**: chỉ dùng open weights, ≤ 15B tham số, phát hành trước 01/08/2026. Không gọi LLM đóng (OpenAI, Gemini, …) trong pipeline sinh kết quả. Ghi tên repo HF và revision của model trong config.
5. **Dữ liệu**: không commit `data/`, `artifacts/`, `runs/` (đã có trong `.gitignore`) hoặc file lớn hơn vài MB. Mọi nguồn dữ liệu ngoài phải ghi lại (tên, URL, license) để trích dẫn trong working notes.
6. **Bí mật**: `HF_TOKEN`, `NCBI_API_KEY`, … chỉ đọc từ biến môi trường. Không ghi vào code, config hay commit.
7. **Crawl lịch sự**: giới hạn tốc độ theo domain, retry có backoff, User-Agent trung thực, tôn trọng robots.txt. Không vượt CAPTCHA hay cơ chế chống bot; trang bị chặn thì dùng nguồn dự phòng như mô tả trong `docs/data_ingestion.md`.

## 5. Quy ước code

- Code, tên biến, comment và docstring viết tiếng Anh; tài liệu trong `docs/` viết tiếng Việt.
- Theo phong cách sẵn có: `from __future__ import annotations`, type hint đầy đủ, `pathlib.Path`, đọc/ghi file với `encoding="utf-8"` và `ensure_ascii=False` cho JSON.
- Ruff, line length 100, target py310 (`pip install -e '.[dev]'` rồi `ruff check src tests`).
- Không thêm dependency nặng vào core. Dependency mới đặt vào optional extra trong `pyproject.toml`.
- Test dùng `unittest`, dữ liệu test nhỏ, không gọi mạng và không tải model. Code mạng/model cần một lớp mỏng để test được bằng dữ liệu giả.
- Các job dài (crawl, encode, index) phải resume được và có log tiến độ.

## 6. Config và thí nghiệm

- Config thí nghiệm nằm ở `configs/experiments/NN_ten_ngan.yaml`, kế thừa `../base.yaml`. Dùng `_replace_: true` khi muốn thay hẳn một block thay vì merge.
- Config đã dùng để nộp leaderboard thì không sửa nữa; muốn đổi thì copy sang số mới.
- Mỗi run lưu ở `runs/<YYYYMMDD>_<ten>/` gồm config đã resolve (`vibiomir show-config`), git commit, metric, thời gian chạy.
- Mỗi lần nộp leaderboard ghi một dòng vào `docs/submission_log.md` (tạo nếu chưa có): thời gian, người nộp, config, commit, các điểm số, ghi chú.

## 7. Git

- `main`: bản ổn định. `develop`: nhánh tích hợp. Việc mới làm trên nhánh `feat/<ticket>-<mo-ta-ngan>` (ví dụ `feat/D2-pubmed-eutils`) tách từ `develop`, mở PR vào `develop`.
- Commit nhỏ, mỗi commit một ý, message tiếng Anh theo Conventional Commits: `feat(ingest): add PubMed E-utilities fetcher`, `fix(evaluation): macro-average per-query F2`, `docs: …`, `test: …`, `chore: …`.
- **Không thêm trailer `Co-Authored-By`, dòng "Generated with …" hoặc bất kỳ ghi chú AI nào vào commit message hay mô tả PR.**
- Chạy test trước khi commit. Không force-push lên `main` hoặc `develop`.
- CI (`.github/workflows/ci.yml`, GitHub Actions) chạy trên mọi PR và push vào `develop`/`main`, với Python 3.10 và 3.13: cài `.[ingest,dev]`, chạy `ruff check src tests` rồi `bash scripts/smoke_all.sh` (unit test, cú pháp script, factory của `configs/full`, smoke Qdrant trên `examples/`). CI phải xanh trước khi merge. CI không có GPU, không có secret, và test không được gọi mạng hay tải model (mục 5).
- `MANIFEST.sha256` là bản ghi hash của codebase ban đầu (commit `af2eec9`); không có code nào đọc file này. Đừng sửa tay.

## 8. Phân vai

| Người | Phụ trách | Ticket |
|---|---|---|
| Hoàng | Lead, crawl/ingest corpus, lịch nộp bài | D*, điều phối |
| Hiệp | Data cùng Hoàng (PubMed, nguồn bootstrap, QA), evaluator, đóng gói và nhật ký nộp bài, dev set | D2, D3, E* |
| Hưng | Retrieval và model: index full corpus, dense, rerank, cutoff; working notes | R*, P* |

Khi nhận việc, agent đọc ticket tương ứng trong `docs/plan.md`, làm đúng phạm vi ticket, rồi báo lại: đã làm gì, chạy lệnh gì để kiểm tra, còn gì dở dang.

## 9. Một việc được coi là xong khi

- Test pass và có test mới cho logic mới.
- Smoke test trên `examples/` vẫn chạy.
- Không phá bất biến ở mục 4.
- Doc liên quan đã được cập nhật (`docs/plan.md`: đánh dấu trạng thái ticket).
