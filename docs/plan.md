# Kế hoạch thực thi R2AI2026 (25/10 → 11/11/2026)

Bản lập ngày Chủ nhật 25/10/2026. Luật thi và số liệu nằm trong [`competition.md`](competition.md); thiết kế crawl nằm trong [`data_ingestion.md`](data_ingestion.md). Mỗi ticket có mã (D/E/R/P) để agent nhận việc; cập nhật cột trạng thái khi làm xong.

## 0. Tóm tắt

- **Mục tiêu**: FINAL ≥ 0,50 trên public trước 31/10 23:59 (đội đứng đầu hiện đạt 0,487) và giữ được mức đó ở private.
- **Đường đi**:
  1. Trong 48 giờ có corpus đủ ~1,22M doc: PubMed qua API, cộng bản crawl công khai trên HF, rồi tự crawl phần còn thiếu.
  2. BM25 trên toàn corpus, chỉ chọn ít doc (giữ precision cao như baseline của BTC), chunk có độ dài sát chunk vàng (luật mới đòi LCS ≥ 40% cả độ dài chunk vàng lẫn chunk nộp) → nộp bài đầu tiên ngày 26/10.
  3. Thêm nhánh xuyên ngữ (dịch query + dense đa ngữ) và reranker.
  4. Tune số doc/chunk mỗi query bằng dev set và leaderboard.
- **Việc phải làm ngay hôm nay**: xin quyền push repo, chốt tài khoản nộp bài của đội, tải dữ liệu chính thức, sửa evaluator cho khớp luật.

## 1. Hiện trạng (25/10)

| Mảng | Tình trạng |
|---|---|
| Repo | Framework ablation đủ module (chunk, query, retrieval, fusion, rerank, aggregation, selection), nhưng chỉ chạy in-memory trên dữ liệu mẫu. Chưa có crawl/ingest, chưa có runner cho toàn corpus. |
| Evaluator | Lệch luật ở ba điểm: ghép chunk 1–1; tính F2 từ P/R trung bình thay vì trung bình F2 từng query; chỉ xét LCS/\|g\|, thiếu điều kiện LCS/\|c\| ≥ 0,4 mà BTC bổ sung ngày 25/10 (`competition.md` §3). |
| Dữ liệu | Chưa tải. BTC chỉ phát URL. Đã có ít nhất hai bản crawl công khai trên HF map đúng id chính thức (`data_ingestion.md` §2.2). |
| Máy local | i5-13500H, 16 GB RAM, không GPU NVIDIA, ổ D còn ~26 GB trống, Java 17. Đủ cho crawl và BM25; encode embedding và rerank phải chạy GPU ngoài (Kaggle 2×T4 30 giờ/tuần/tài khoản, Colab hoặc máy thuê). **Cập nhật 28/10**: chuyển sang laptop Linux RTX 5060 8 GB (Java 21, ~30 GB disk trống), encode/rerank chạy local với cap 40% VRAM (`docs/automation.md`); toàn bộ chuỗi crawl→index→submission chạy tự động qua `scripts/auto_pipeline.py`. |
| GitHub | Tài khoản `HuyHoangTran05` chỉ có quyền read trên `HungPhamNoob/thh`, nên Hưng cần thêm làm collaborator. |
| Leaderboard | Top 1: 0,4871. Baseline BM25 của BTC: 0,3764, nhưng doc F2 0,5587 (cao nhất bảng). Các điểm này chấm theo luật chunk cũ; từ 25/10 kết quả chấm chỉ còn điểm tổng, không còn chi tiết theo query. |

## 2. Phân vai

| Người | Vai trò | Ticket |
|---|---|---|
| Hoàng | Lead: crawl/ingest, điều phối, quyết định bài nộp | P0, D0, D1, D4, D5, D6, D7 |
| Hiệp | Data cùng Hoàng (PubMed, bootstrap, QA); evaluator, đóng gói, dev set, nhật ký nộp bài | D2, D3, D8, E1–E5 |
| Hưng | Retrieval và model; GPU jobs; working notes | R0–R8, P1, P2 |

Agent (Codex/Claude) làm từng ticket trên nhánh `feat/<ticket>-…`. Con người review PR, chạy job dài và nộp bài.

## 3. Lịch theo ngày (giờ VN)

| Ngày | Data | Eval / nộp bài | Retrieval | Nộp |
|---|---|---|---|---|
| CN 25/10 | P0; D0, D1; D3 tải bootstrap + báo cáo độ phủ; D2 bắt đầu | E1 | R0 thiết kế + cài Java 21/Pyserini | – |
| T2 26/10 | D2 xong; D6 corpus **v0** (bootstrap + PubMed); D4 crawler chạy thử | E2, E4 | R0 BM25 doc-level trên corpus v0; chunk trong doc ứng viên | **#1 trước 22:00** (mục tiêu ≥ 0,40) |
| T3 27/10 | D4 crawl phần thiếu tiếng Việt (domain lớn trước); D5 extractor + QA | E3 dev set (silver + 50 query gán tay) | R1 xử lý query dài; R6 cutoff doc/chunk + độ dài chunk | 2–3 bài |
| T4 28/10 | D4 crawl phần thiếu tiếng Trung; D7 nguồn dự phòng | E5 phân tích lỗi theo ngôn ngữ | R3 dịch VI→EN/ZH + BM25 EN/ZH; R4 encode dense doc-level trên GPU | 2–3 bài |
| T5 29/10 | D6 corpus **v1**, đóng băng lúc 18:00 | tune trên dev | R4 dense + fusion (R7); R5 reranker | 2–3 bài |
| T6 30/10 | D8 doc store cho v1 | chọn ứng viên | re-index v1; tune R5/R6 | 2–3 bài |
| T7 31/10 | – | **nộp public cuối trước 23:59**; chốt 2–3 config cho private | đóng băng code | phần còn lại |
| 01–03/11 | – | **private: tổng 5 bài**, hết hạn 03/11 23:59 | chỉ sửa lỗi | ≤ 5 |
| 04/11 | – | nộp hệ thống (P2) | – | – |
| 04/11 → hạn BTC | – | – | P1 working notes | – |

## 4. Backlog

Trạng thái: `todo` / `doing` / `done`. Ưu tiên: P0 > P1 > P2.

### P – quản lý

| Mã | Ưu tiên | Việc | Tiêu chí xong | Trạng thái |
|---|---|---|---|---|
| P0 | P0 | (1) Hưng thêm `HuyHoangTran05` làm collaborator repo. (2) Xác nhận đội đã đăng ký competition 15 và chốt **một** tài khoản nộp bài. (3) Gửi BTC 5 câu hỏi ở `competition.md` §8. (4) Đăng ký NCBI API key và HF token (chỉ đặt trong biến môi trường). | Push được nhánh `develop`; biết ai nộp bài; email đã gửi | todo |
| P1 | P1 | Working notes paper: dàn ý (data, pipeline, thí nghiệm, kết quả, nguồn dữ liệu ngoài), viết dần từ 01/11. | Bản nháp trước hạn BTC | todo |
| P2 | P0 | Gói nộp hệ thống 04/11: README tái lập, config cuối, danh sách model (repo HF + revision + ngày phát hành + số tham số), danh sách nguồn dữ liệu và license. | Người ngoài chạy lại được từ README | todo |

### D – dữ liệu (chi tiết trong `data_ingestion.md`)

| Mã | Ưu tiên | Việc | Tiêu chí xong | Trạng thái |
|---|---|---|---|---|
| D0 | P0 | Lệnh `vibiomir ingest download-official`: tải `query.parquet` + `links_corpus_vi_en_zh.parquet` về `data/raw/official/`. | 1.200 query, 1.220.598 link, id unique | done (25/10) |
| D1 | P0 | `ingest/domains.py`: bảng domain → ngôn ngữ (94 domain, gồm cả domain Trung có tên "lạ") + manifest `id,url,domain,lang`. | Đếm đúng vi 740.598 / en 240.000 / zh 240.000; có unit test | done (25/10): `vibiomir ingest manifest` đếm đúng |
| D2 | P0 | PubMed qua E-utilities (lô 200 PMID, ≤ 10 req/s với API key, resume được) → title + abstract. | ≥ 99% PMID có title; báo cáo tỉ lệ có abstract | doing (25/10): `vibiomir ingest pubmed` đang chạy; còn báo cáo tỉ lệ có abstract |
| D3 | P0 | Nhập bản crawl công khai (`lighth0406/R2AI_STAGE3_ds`, `tmh1008/ViBioMIR-crawled-corpus`) về schema chung, kèm cột `source`. Báo cáo độ phủ và chất lượng theo domain. | `data/processed/coverage.csv`; mỗi domain có 5 mẫu đã so với trang gốc | doing (25/10): 25/10 quyết định **không** đưa bản công khai vào corpus nộp bài khi chưa có xác nhận của BTC; `ingest bootstrap` chỉ dùng để so chất lượng |
| D4 | P0 | Crawler bất đồng bộ: SQLite state, rate limit theo domain, retry/backoff, nhận diện charset (gb18030), robots.txt, chạy được lại sau khi bị ngắt. | Crawl 1.000 URL rải trên 12 domain lớn, ≥ 95% thành công ở domain truy cập được; kill rồi chạy lại không lấy trùng | doing: code + test xong, pilot 25/10 đạt; đang tự crawl toàn bộ (xem `data_ingestion.md` §2.5–2.6) |
| D5 | P0 | Extractor: trafilatura mặc định + extractor riêng cho 8 domain Việt và 4 nhóm domain Trung lớn nhất; golden test HTML; QA 20 trang/domain. | Text đúng nội dung chính, đoạn tách bằng xuống dòng, không dính token | doing: rule theo site + rule CMS chung + trafilatura; còn QA 20 trang/domain |
| D6 | P0 | Merge theo thứ tự ưu tiên nguồn → `data/processed/corpus/` + coverage. v0 xong 26/10, v1 đóng băng 29/10 18:00. | Đúng 1.220.598 dòng, id unique, báo cáo doc rỗng theo domain | doing: v0 xong 26/10 09:24 (76,5% id có text), v1 chờ Wayback |
| D7 | P1 | Nguồn dự phòng cho trang chết hoặc bị chặn: Wayback `id_`; Playwright cho domain chỉ render bằng JS. Không vượt CAPTCHA/anti-bot. | Số doc rỗng giảm, ghi rõ `source` | doing: job Wayback chạy từ 26/10 09:21, ~25–40 giờ |
| D8 | P0 | Doc store (SQLite hoặc parquet có chỉ mục) để lấy text theo id; xuất JSONL cho subset. | Lấy 1.000 doc ngẫu nhiên < 1 s; RAM < 1 GB | todo |

### E – đánh giá và nộp bài

| Mã | Ưu tiên | Việc | Tiêu chí xong | Trạng thái |
|---|---|---|---|---|
| E1 | P0 | Sửa `evaluation/official.py` cho khớp luật: cặp (c, g) khớp khi **cả** LCS/\|g\| ≥ 0,4 **và** LCS/\|c\| ≥ 0,4; chunk nộp đúng nếu khớp **bất kỳ** chunk vàng nào (không ghép 1–1); recall = số chunk vàng có ít nhất một chunk nộp khớp; F2 tính theo từng query rồi trung bình; `final_score = (doc_f2 + chunk_f2)/2`. Sửa `test_one_to_one_chunk_matching` theo luật mới; thêm test có số liệu tính tay, gồm ca chunk nộp quá dài (chứa trọn chunk vàng nhưng dài hơn 2,5 lần) phải bị tính sai. | Test mới pass; README/docs ghi đúng tên metric | done |
| E2 | P0 | `vibiomir package`: validate (đủ 1.200 id, id là int, doc id thuộc corpus, `chunk_text` là substring, không rỗng), tuỳ chọn kiểu doc id `int\|str`, nén ZIP **một file ở gốc**. | ZIP mở ra đúng 1 file; validator bắt được lỗi cố ý | done (28/10): `vibiomir package --submission … --corpus … --output …` validate streaming cả full corpus (không nạp hết vào RAM) rồi nén ZIP 1 file ở gốc; validate với lỗi cố ý có test (`tests/test_package.py`). Chưa làm cờ `int\|str` vì writer đã chuẩn hoá id số. |
| E3 | P0 | Dev set: (a) silver từ trang hỏi đáp chứa gần nguyên văn query (BM25 điểm vượt trội, người kiểm tra lại); (b) 50–100 query gán tay doc + chunk bằng script gợi ý ứng viên; (c) tuỳ chọn: query tổng hợp kiểu GPL bằng LLM mở. Định dạng giống `examples/gold.json`. | ≥ 100 query có nhãn, ghi rõ nguồn từng nhãn | todo |
| E4 | P0 | Nhật ký: `docs/submission_log.md` (mỗi lần nộp một dòng) + thư mục `runs/<ngày>_<tên>/` lưu config đã resolve, commit, metric. | Mọi bài nộp đều tra ngược được ra config + commit | doing (28/10): đã tạo `docs/submission_log.md`; auto pipeline tự ghi `run_info.json` (commit, config) + `config_resolved.yaml` cạnh mỗi submission |
| E5 | P1 | Phân tích lỗi: tách theo ngôn ngữ doc, độ dài query; phân loại không có kết quả / sai doc / đúng doc sai chunk / chunk quá ngắn. | Báo cáo ngắn sau mỗi thay đổi lớn | todo |

### R – retrieval

| Mã | Ưu tiên | Việc | Tiêu chí xong | Trạng thái |
|---|---|---|---|---|
| R0 | P0 | Runner cho toàn corpus, không nạp hết vào RAM: local SQLite artifact store + persistent Qdrant/Lucene/PLAID indexes, chunk-first hoặc cascade, selection → submission. | Code và smoke test xong; benchmark 1.200 query/RAM trên corpus v1 còn chờ artifact thật | implemented; 28/10: chuỗi crawl→merge→store→Lucene/Qdrant→run→package chạy unattended qua `scripts/auto_pipeline.py` (`docs/automation.md`), benchmark chờ artifact thật |
| R1 | P0 | Query dài nhiều ý: tách câu/ý, rút keyphrase y khoa (LLM mở, ví dụ Qwen2.5-7B-Instruct, chế độ `keywords` đã có), giữ query gốc làm một variant. | Cải thiện trên dev; không giảm doc precision | todo |
| R2 | P1 | Nhận diện trang nguồn gần trùng query (trang hỏi đáp chứa câu hỏi): điểm BM25 top-1 vượt xa top-2 → coi là doc chắc chắn. | Đo precision của luật trên dev | todo |
| R3 | P0 | Xuyên ngữ: dịch query VI→EN/ZH (NLLB-200 hoặc LLM mở ≤ 15B), BM25 trên index EN (PubMed) và ZH (bigram CJK); tuỳ chọn MedCPT cho nhánh EN. | Recall doc EN/ZH tăng trên dev | todo |
| R4 | P1 | Dense đa ngữ doc-level: Qwen3-Embedding-0.6B hoặc BGE-M3 trên title + ~512 token đầu, encode trên GPU, Qdrant; dense cấp chunk chỉ tính cho doc ứng viên khi query. | Index 1,22M doc; tăng recall@100 trên dev | todo |
| R5 | P1 | Rerank chunk của top doc: bge-reranker-v2-m3 / Qwen3-Reranker-0.6B/4B / gte-multilingual-reranker-base. | Tăng chunk F2 trên dev; 1.200 query < 1 giờ trên T4 | todo |
| R6 | P0 | Selection: số doc mỗi query (ngưỡng tương đối + gap, có thể theo ngôn ngữ), số chunk mỗi doc, và **độ dài chunk**. Theo luật mới, chunk chỉ được tính khi dài khoảng 0,4–2,5 lần chunk vàng, nên phải ước lượng cỡ chunk vàng: 2–3 bài nộp giữ nguyên tập doc, chỉ đổi cỡ chunk (1 câu / 1 đoạn / 2–3 đoạn), so chunk F2; tiếng Trung cắt theo đoạn. Sweep phần còn lại trên dev. | Bảng sweep, cỡ chunk được chọn có bằng chứng từ leaderboard | todo |
| R7 | P1 | Fusion nhiều retriever và ngôn ngữ (RRF có trọng số), hiệu chỉnh trọng số trên dev. | Tốt hơn retriever đơn tốt nhất | todo |
| R8 | P2 | Chỉ làm nếu còn thời gian: fine-tune embedding/reranker bằng dữ liệu tổng hợp (GPL/InPars, `training_recipes.md`). | – | todo |

Model ứng viên phải được kiểm tra trên model card: phát hành trước 01/08/2026, ≤ 15B tham số, có trọng số công khai. Các model đã có trong repo (BGE-M3, mGTE, MedCPT, BMRetriever, NLLB-200, Qwen2.5-7B-Instruct, bge-reranker-v2-m3) đều phát hành trước 2026, nhưng vẫn cần ghi lại revision.

## 5. Chiến lược dùng lượt nộp

- Public còn 7 ngày × 10 bài. Mỗi thay đổi lớn nộp ít nhất 1 bài; sweep nhỏ làm trên dev set. Không tune cutoff đến từng chữ số theo leaderboard public.
- Bài #1 (26/10) chỉ cần đúng format và corpus đủ. Dùng nó để xác nhận kiểu `doc_id` (int hay str) và mức điểm của BM25 trên corpus của mình.
- BTC đã tắt kết quả chấm chi tiết, mỗi bài chỉ trả về điểm tổng. Vì vậy mỗi bài nộp nên trả lời đúng một câu hỏi, tức chỉ đổi một yếu tố so với bài trước (ví dụ chỉ đổi cỡ chunk ở R6), để đọc được tác động từ điểm tổng.
- Private chỉ có 5 bài cho cả phase: ngày 01/11 nộp ngay config tốt nhất từ public; 1–2 biến thể có lý do rõ ràng; giữ 2 bài dự phòng cho sửa lỗi. Vì cùng 1.200 query, phân phối private giống public, nên ưu tiên config ổn định.

## 6. Rủi ro và cách xử lý

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Crawl chậm hoặc bị chặn | Thiếu doc → mất recall | Bootstrap từ HF trước, crawl phần thiếu theo thứ tự domain lớn; nguồn dự phòng (D7); chấp nhận thiếu ở domain nhỏ |
| Text crawl khác bản BTC | Chunk lệch → overlap < 0,4 | Extractor giữ nguyên đoạn; cắt chunk theo ranh giới đoạn; QA theo domain |
| Cỡ chunk lệch chunk vàng (luật mới, LCS/\|c\| ≥ 0,4) | Trúng đúng chỗ nhưng vẫn bị tính sai | Ước lượng cỡ chunk vàng bằng các bài nộp chỉ đổi cỡ chunk (R6); không gộp nhiều đoạn vào một chunk khi chưa có bằng chứng |
| Token LCS cho tiếng Trung chưa rõ | Chunk ZH có thể luôn trượt | Cắt chunk ZH theo đoạn; hỏi BTC; thử 1 bài nộp chỉ thay đổi nhánh ZH |
| Bản crawl cộng đồng có thể không được chấp nhận | Phải crawl lại | Hỏi BTC sớm; trích dẫn đầy đủ; ưu tiên tự crawl domain lớn |
| RAM 16 GB, không GPU | Không chạy được toàn corpus in-memory, không encode được | R0 dùng index offline + doc store; GPU ngoài cho R4/R5 |
| Hết lượt private | Không sửa được lỗi | Validate + package bắt buộc (E2); giữ 2 lượt dự phòng |
| Không push được repo | Team không đồng bộ | P0: thêm collaborator; tạm thời làm việc trên fork |
| Thiếu working notes | Kết quả không được công nhận | P1 bắt đầu từ 01/11, tận dụng `docs/` |

## 7. Một bài nộp hợp lệ khi

1. `vibiomir validate --submission … --corpus …` không báo lỗi.
2. ZIP chứa đúng một file JSON ở gốc.
3. Đã ghi vào `docs/submission_log.md`: config, commit, người nộp, thời gian.
4. Config dùng để nộp đã commit và không sửa lại.
