# Thiết kế crawl/ingest corpus

Phần khó nhất của cuộc thi: biến 1.220.598 URL thành corpus text sạch, đúng id, trong vài ngày, trên một laptop 16 GB RAM, ổ D còn ~26 GB. Ticket tương ứng: D0–D8 trong [`plan.md`](plan.md). Số liệu đo ngày 25/10/2026.

## 1. Đầu ra

`data/processed/corpus/part-*.parquet`, mỗi id chính thức đúng một dòng:

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `id` | int64 | id trong `links_corpus` |
| `url`, `final_url` | str | URL gốc và URL sau redirect |
| `domain`, `language` | str | `language` ∈ `vi`/`en`/`zh`, lấy từ bảng domain (D1), không đoán bằng langdetect |
| `title`, `text` | str | nội dung chính; các đoạn cách nhau bằng `\n\n` |
| `source` | str | `crawl`, `pubmed_eutils`, `hf_lighth0406`, `hf_tmh1008`, `wayback`, `missing` |
| `extractor` | str | `trafilatura`, `site:<domain>`, `pubmed_xml`, … |
| `http_status`, `fetched_at`, `published_at` | | nếu có |
| `text_sha256`, `raw_sha256` | str | để dedup và truy vết |

Kèm theo:

- `data/processed/coverage.csv`: theo domain, gồm tổng số link, số doc OK, ngắn (< 200 ký tự), lỗi, và số doc theo từng nguồn.
- Lệnh xuất JSONL cho subset, đúng định dạng `vibiomir.io.load_documents` đọc được (`id`, `url`, `title`, `language`, `text`, các trường còn lại vào `metadata`).

Yêu cầu về text:

- Chỉ giữ nội dung chính: tiêu đề và thân bài. Với trang hỏi đáp, giữ câu hỏi và câu trả lời của bác sĩ. Bỏ menu, footer, "bài liên quan", quảng cáo, bình luận.
- Giữ nguyên ký tự gốc: không bỏ dấu, không dịch, không viết lại, không chuẩn hoá Unicode khi lưu (evaluator tự chuẩn hoá NFKC khi chấm).
- Mọi phần tử khối (`p`, `li`, `div`, `br`, `td`, …) phải được ngăn cách bằng khoảng trắng hoặc xuống dòng. Nếu `<p>A</p><p>B</p>` thành `AB`, token LCS sẽ sai.
- Tiếng Trung: giữ nguyên từng đoạn như trên trang, không thêm hoặc bớt ký tự bên trong đoạn. Lý do: xem `competition.md` §3 ý 4.

## 2. Nguồn theo ngôn ngữ

| Ngôn ngữ | Link | Cách lấy chính |
|---|---:|---|
| en | 240.000 | NCBI E-utilities (không crawl HTML) |
| vi | 740.598 | bản crawl công khai trên HF, sau đó tự crawl phần thiếu |
| zh | 240.000 | tự crawl là chính (bản công khai phủ ít), sau đó nguồn dự phòng |

### 2.1 Tiếng Anh: PubMed (D2)

- PMID lấy từ URL `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`.
- `efetch.fcgi?db=pubmed&retmode=xml&id=<pmid,…>`, mỗi lô 200 PMID (lô lớn hơn dùng POST), kèm `api_key`, `tool`, `email` theo chính sách NCBI. Có key: ≤ 10 request/giây; không có key: ≤ 3 request/giây. 240.000 / 200 = 1.200 request, chạy hết trong vài phút đến vài chục phút.
- `text` = `ArticleTitle` + `\n\n` + các đoạn `AbstractText`. Abstract có cấu trúc giữ nhãn (`BACKGROUND: …`). Keyword/MeSH lưu vào metadata; có thể đưa vào `index_text`, không đưa vào `text`.
- Bài không có abstract (thường là bài cũ) chỉ có title; đánh dấu `short` trong coverage.
- Lưu XML gốc theo lô (nén gz) để parse lại khi cần mà không phải gọi API lại.

### 2.2 Bản crawl công khai trên Hugging Face (D3)

| Dataset | Quy mô | Nội dung | Độ phủ đã đo |
|---|---|---|---|
| [`lighth0406/R2AI_STAGE3_ds`](https://huggingface.co/datasets/lighth0406/R2AI_STAGE3_ds) | `corpus.parquet` 963 MB, 610.949 dòng; thêm thư mục `runs/` có `missing_links`, `short_text_links` | `id` chính thức, `url`, `final_url`, `title`, `text`, `extractor` (trafilatura + vài extractor riêng cho 120ask, cnkang, bachmai), `published_at`, `text_sha256`, `duplicate_doc_ids`, cột truy vết `vibiomir_*` (100% `exact_id_url`) | vi **560.380 / 740.598 (75,7%)**, zh 50.569 / 240.000 (21,1%), en 0 |
| [`tmh1008/ViBioMIR-crawled-corpus`](https://huggingface.co/datasets/tmh1008/ViBioMIR-crawled-corpus) | ~1.000 shard, 1,9 GB, 1.174.600 dòng | `id`, `url`, `domain`, `status_code`, `success`, `title`, `content`, `content_length` | 878.943 dòng `success=True`; chưa tách theo ngôn ngữ (HF trả 429 khi đọc song song nhiều shard, cần tải tuần tự) |
| `522H0134-NguyenNhatHuy/ViBioMIR-Corpus` | 4,2 GB | gated, phải xin quyền | chưa kiểm tra |

Cách dùng:

1. Tải bằng `huggingface_hub.snapshot_download` (giới hạn `allow_patterns`, tải tuần tự để tránh 429), lưu vào `data/raw/hf/<dataset>/`.
2. Chuyển về schema ở §1, đặt `source=hf_<tên>`.
3. Mỗi domain lấy 5–20 doc, so với trang thật trên trình duyệt; chọn nguồn tốt nhất cho từng domain.
4. Ghi nguồn và license vào danh sách trích dẫn cho working notes; hỏi BTC xem có được dùng không (`competition.md` §8 câu 5). Nếu BTC không cho dùng, các nguồn này chỉ dùng để đối chiếu chất lượng, còn corpus nộp bài lấy từ tự crawl.

Không coi bản công khai là chuẩn: domain nào có text thiếu hoặc lỗi thì tự crawl lại.

### 2.3 Tự crawl tiếng Việt và tiếng Trung (D4)

```text
links parquet ─► manifest (id, url, domain, lang) ─► crawl_state.sqlite
                                                        │
                                  async fetcher (httpx, rate limit theo domain)
                                                        │  raw HTML gz (chỉ giữ mẫu + trang lỗi)
                                  extractor registry (riêng theo site → trafilatura)
                                                        │
                                  parts/<source>/part-*.parquet ─► merge + QA ─► corpus/
```

- **State** (SQLite): bảng `pages(id PK, url, domain, lang, status, attempts, http_status, final_url, error, fetched_at, text_sha256, source)`, với `status` ∈ `pending`/`ok`/`short`/`failed`/`blocked`/`gone`. Khi chạy lại chỉ lấy `pending` và `failed` còn lượt thử. Ghi theo lô để không khoá DB.
- **Fetcher**: asyncio + httpx, dùng chung connection pool. Khoảng 128 kết nối toàn cục, mỗi domain 2–8 kết nối. Mỗi domain một token bucket bắt đầu ở 2 request/giây; tăng dần khi ổn định, giảm một nửa khi gặp 429/503. Timeout 20 giây, retry 3 lần với backoff mũ có jitter, theo redirect và lưu `final_url`. User-Agent ghi tên dự án và email liên hệ; đọc robots.txt của từng domain.
- **Lập lịch**: lấy URL xen kẽ giữa các domain (round-robin) để 50+ domain chạy song song. Throughput đến từ số domain chạy cùng lúc, không phải từ việc ép một domain chạy nhanh.
- **Ước lượng**: 50 domain × 2 request/giây ≈ 100 trang/giây trên lý thuyết. Nút thắt là domain lớn: ví dụ 120ask.com có 98.160 link, ở 4 request/giây mất khoảng 7 giờ. Vì vậy phải lấp từ bootstrap trước và chỉ crawl phần còn thiếu.
- **Charset**: lấy từ header/meta, dự phòng bằng `charset-normalizer`. Trang Trung khai báo gbk/gb2312 thì decode bằng `gb18030`.
- **Raw HTML**: ổ đĩa không đủ để giữ toàn bộ (ước tính 15–20 GB dạng gz). Chỉ giữ 200 trang mẫu mỗi domain và mọi trang extract lỗi hoặc ngắn, để debug extractor.
- **Trang bị chặn** (403/429 kéo dài, 521, trang thử thách JS): đánh dấu `blocked`, không tìm cách vượt. Thứ tự dự phòng (D7): bản crawl công khai → Wayback Machine (`https://web.archive.org/web/2026id_/<url>`, ghi `source=wayback`) → để trống và ghi vào coverage.
- **Trang render bằng JS**: chỉ khi thử 20 trang của domain đó thấy httpx không lấy được nội dung mới dùng Playwright headless, với rate limit thấp.

### 2.4 Extractor (D5)

- Mặc định dùng trafilatura (bỏ comment, giữ bảng, ưu tiên recall), xuất text có ngắt đoạn.
- Viết extractor riêng (CSS selector cho title và thân bài, bỏ khối liên quan/quảng cáo) cho các domain lớn, đăng ký vào registry theo domain:
  - **Việt** (8 domain, chiếm 68% link tiếng Việt): suckhoecongdongonline.vn 154.503, suckhoedoisong.vn 85.823, nhathuoclongchau.com.vn 79.598, thanhnien.vn 72.111, laodong.vn 31.282, phunusuckhoe.giadinhonline.vn 27.577, vinmec.com 26.129, medlatec.vn 24.762.
  - **Trung** (4 nhóm, chiếm 88% link tiếng Trung): 120ask.com 98.160 (hỏi + đáp), *.39.net 47.261 (ask.39.net là hỏi đáp, còn lại là bài viết theo chuyên mục), cnkang.com 40.046, familydoctor.com.cn 26.819.
- Golden test: mỗi extractor có 2–3 file HTML mẫu đã cắt gọn trong `tests/fixtures/html/<domain>/`, kèm đoạn text bắt buộc phải có và đoạn bắt buộc không được có.
- QA: mỗi domain lấy ngẫu nhiên 20 trang so với trình duyệt; đo tỉ lệ text < 200 ký tự; nhận diện trang lỗi trá hình (404 mềm, "Trang không tồn tại", trang thử thách bot) bằng pattern.

### 2.5 Triển khai D4/D5 và số đo pilot (25/10)

Code nằm ở `src/vibiomir/ingest/` (`official.py`, `domains.py`, `state.py`, `fetch.py`, `extract.py`, `cli.py`); cài `pip install -e '.[ingest]'`.

```bash
vibiomir ingest download-official                 # data/raw/official/
vibiomir ingest manifest                          # data/interim/manifest.parquet
vibiomir ingest crawl --limit-per-domain 30       # pilot; bỏ cờ này để crawl hết
vibiomir ingest crawl --exclude-ids data/interim/bootstrap/lighth0406_covered_ids.parquet  # chỉ phần thiếu
vibiomir ingest status                            # đếm theo ngôn ngữ / domain / trạng thái
vibiomir ingest reset --domain 39.net --status short   # đưa trang về pending sau khi sửa extractor
```

- Dừng bằng Ctrl+C rồi chạy lại cùng lệnh là tiếp tục: state ở `data/interim/crawl_state.sqlite`, text ở `data/interim/parts/crawl/part-*.parquet` (schema §1, thêm cột `encoding`), log ở `data/interim/logs/`. Trang `failed`/`blocked` được thử lại ở lần chạy sau cho tới `--max-attempts` (mặc định 3). Một id có thể xuất hiện ở nhiều part (crawl lại sau `reset`); D6 giữ dòng có `fetched_at` mới nhất.
- Trạng thái: `pending`, `ok`, `short` (< 200 ký tự ngoài title), `failed`, `blocked`, `gone` (404/410 hoặc 404 mềm), `disallowed` (robots.txt cấm), `unsupported` (không phải HTML: pdf, pptx, ảnh…; không tính vào bộ ngắt domain).
- Rate limit tính theo **domain đăng ký** (`*.39.net` dùng chung một giới hạn, `ask.39.net` tách riêng): bắt đầu 2 request/giây, cộng 0,5 sau mỗi 50 lần thành công, trần 6; gặp 429/503 thì giảm một nửa. Lý do: pilot đầu giới hạn theo host, ~20 subdomain 39.net cộng lại thành ~40 request/giây và bị chuyển sang captcha `image.39.net/verify.html`. Sau khi gộp nhóm, 627/630 trang 39.net ok.
- Ngắt theo domain: 10 phản hồi chặn/lỗi liên tiếp mà chưa có trang nào thành công (hoặc 50 liên tiếp) thì dừng domain đó cho lần chạy hiện tại. Trang captcha nhận diện qua mẫu nội dung, qua URL đích (`/verify.html`, …) và qua cùng một text ngắn lặp lại ≥ 5 lần trong một domain.
- Extractor: rule riêng (bachmai, vinmec, medlatec, giadinhonline, youmed, hellobacsi, vietnamnet, ask.39.net, *.39.net, cnkang, familydoctor, mediawiki) → rule CMS chung cho báo Việt (`.detail-sapo/.detail-content`, `.article__sapo/.article__body`, `.content-fck`, `.td-post-content`, `.entry-content`, `[itemprop=articleBody]`) → trafilatura → container nhiều `<p>` nhất. `text` = title + `\n\n` + thân bài. Giải mã: charset header → `<meta>` → utf-8 → charset-normalizer; gb2312/gbk đọc bằng gb18030 (cnkang khai báo gb2312 trong `<meta>`, header không có charset).
- User-Agent: `vibiomir-research-crawler/0.1 (+https://github.com/HungPhamNoob/thh; R2AI2026 ViBioMIR research)`; đặt `VIBIOMIR_CRAWLER_CONTACT` để thêm email liên hệ.

Không crawl được từ mạng dev ngày 25/10 (pilot 30 URL/domain, lúc đầu đi qua Cloudflare WARP; đã kiểm tra lại sau khi tắt WARP). Không vượt các cơ chế này; chuyển sang nguồn dự phòng (D7):

| Domain | Link | Hiện tượng |
|---|---:|---|
| pubmed.ncbi.nlm.nih.gov | 240.000 | trang proof-of-work (HTTP 203) → dùng E-utilities (D2), mặc định bị bỏ qua khi crawl |
| www.120ask.com | 98.160 | qua WARP: connect timeout; không WARP: Cloudflare 521 (origin không trả lời), thử lại sau |
| nhathuoclongchau.com.vn | 79.598 | Cloudflare 403 |
| laodong.vn | 31.282 | trang JS đặt cookie rồi reload |
| baohaiphong.vn, baonghean.vn, baodanang.vn, hanoimoi.vn, mega.vietnamplus.vn | 48.012 | 403 "Access Denied" |
| youmed.vn | 7.189 | 403 với mọi User-Agent có URL liên hệ; không đổi UA để né |
| zysjonline.com | 6.591 | Cloudflare 403 |
| pmc-ecm-healthblog.beta.pharmacity.io | 5.333 | nginx 403 |
| article.iiyi.com, bingli.iiyi.com | 734 | connect timeout / 521 |
| www.baidu.com | 11 | robots.txt cấm |

vov.vn (9.784 link) chỉ chặn IP của WARP; không WARP thì lấy được nhưng robots.txt đặt `Crawl-delay: 20`, nên chạy thành process riêng (`--domain vov.vn`, ~54 giờ) và được bỏ khỏi job chính bằng `--skip-domain`. Mỗi process ghi part có tên riêng (`part-<thời điểm>-<pid>-NNNNN.parquet`), nên có thể chạy nhiều process song song nếu chúng nhận các domain khác nhau.

Còn lại 693.904 link crawl được (tính cả vov.vn: 703.688). Số đo pilot: ~35–40 trang/giây tổng khi ~80 domain chạy song song, giới hạn bởi băng thông (~4,7 MB/s); HTML trung bình 118 KB/trang; extract 2–20 ms/trang (4 process là đủ); text parquet ~2 KB/trang.

Ước lượng thời gian (trần 6 request/giây mỗi nhóm domain, băng thông như trên):

| Phạm vi | Trang | HTML tải về | Nút thắt | Thời gian |
|---|---:|---:|---|---|
| Phần `lighth0406` còn thiếu, bỏ domain bị chặn | 155.837 | ~11 GB | ask.39.net 19.216 trang ở ~2,4 trang/giây (độ trễ 4–13 giây/trang) | ~2,5–3 giờ |
| Toàn bộ domain crawl được | 693.904 | ~132 GB | băng thông (~7,8 giờ) và suckhoecongdongonline.vn 154.503 trang ở 6 request/giây (~7,2 giờ) | ~8–10 giờ |
| PubMed qua E-utilities (D2) | 240.000 | – | 1.200 request, ≤ 10 request/giây có API key | vài phút đến ~20 phút |

### 2.6 Phương án đang chạy: tự crawl toàn bộ (quyết định 25/10)

Để minh bạch với quy chế (`competition.md` §5), corpus nộp bài chỉ gồm dữ liệu đội tự thu thập từ `links_corpus`: crawl HTML + PubMed qua E-utilities. Bản `lighth0406` không đưa vào corpus, và bản tải về đã được xoá khỏi máy ngày 25/10. Nếu BTC xác nhận bằng văn bản là được dùng (kèm trích dẫn) thì mới cân nhắc lấp các domain mình bị chặn. Ba job chạy song song:

```bash
vibiomir ingest pubmed                    # data/interim/parts/pubmed/, XML gốc gz ở data/raw/pubmed/
vibiomir ingest crawl --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn
vibiomir ingest crawl --domain vov.vn --links "" --workers 1     # process riêng, Crawl-delay 20
```

Trên Windows, `powershell -ExecutionPolicy Bypass -File scripts/start_crawl.ps1` chạy cả ba job thành process ẩn, tách khỏi terminal đã khởi động chúng (tạo qua `Win32_Process.Create`), nên đóng terminal hay Claude Code thì job vẫn chạy. Job nào đang chạy thì script không mở thêm bản thứ hai. Log nằm ở `data/interim/logs/{crawl_full,crawl_vov,pubmed}.log`. Script cũng mở `scripts/keep_awake.ps1`: script này gọi `SetThreadExecutionState` để Windows không tự sleep khi máy rảnh, và tự thoát khi không còn job `vibiomir ingest` nào chạy. Gập máy hoặc bấm nút nguồn vẫn làm máy sleep; khi máy dậy, job tự tiếp tục với các request đang dở được retry, nhưng sau khi khởi động lại máy thì phải chạy lại script. Dừng hẳn một job: `Stop-Process` theo PID tìm bằng `Get-CimInstance Win32_Process`.

- Ước lượng lúc bắt đầu (14:56 25/10): còn 620.850 trang crawl được (ngoài PubMed, vov.vn và domain bị chặn). Nút thắt là 39.net (44.038 trang ở ~1,5 request/giây, ~8 giờ, lâu hơn nếu gặp captcha) và suckhoecongdongonline.vn (147.115 trang ở 6 request/giây, ~7 giờ): **~8–12 giờ**. vov.vn chạy riêng ~54 giờ.
- **PubMed**: 1.200 lô × 200 PMID, POST `efetch.fcgi`, ≤ 3 request/giây khi không có `NCBI_API_KEY` (9 khi có). Mỗi lô ~3 MB XML; cả 240.000 bài mất ~15–20 phút. `text` = `ArticleTitle` + các `AbstractText` (giữ nhãn `BACKGROUND: …`), bỏ `OtherAbstract` và copyright. Trạng thái: `ok` khi abstract ≥ 200 ký tự, `short` khi chỉ có title, `gone` khi PMID không có trong kết quả trả về. Chạy lại thì bỏ qua lô đã có file part.
- **`lighth0406/R2AI_STAGE3_ds`**: tài khoản cá nhân trên HF (tên hiển thị "Hải Đăng Lê Võ", thuộc tổ chức `AncientWings`), tạo ngày 13/10/2026, **không có dataset card và license**. Đã tải để đối chiếu (revision `9fe2787f8495b52bcf677ac24d8e25f18a7553d5`, `corpus.parquet` SHA256 `2b5bd99b…468b5`; `hf_hub_download` qua `hf_xet` bị treo nên phải dùng `curl`) rồi xoá ngày 25/10. Lệnh `vibiomir ingest bootstrap --dataset lighth0406 --input …` (chỉ dùng để đối chiếu) chỉ giữ những dòng có cặp `id`/`url` khớp file chính thức, giữ nguyên `text`, gắn `source=hf_lighth0406`, `extractor=lighth0406:<extractor gốc>`. Cần BTC đồng ý trước khi đưa vào corpus nộp bài (`competition.md` §8 câu 5).

## 3. Merge và kiểm tra (D6)

Thứ tự ưu tiên nguồn cho mỗi id: tự crawl (`ok`) > PubMed E-utilities > bản công khai tốt nhất cho domain đó > Wayback > rỗng.

- Giữ đủ mọi id, kể cả khi trùng text: nhãn chấm theo id. `text_sha256` chỉ dùng để encode embedding một lần cho các bản trùng.
- Kiểm tra cuối: đúng 1.220.598 dòng, id unique, ngôn ngữ khớp bảng domain, tỉ lệ doc rỗng theo domain, phân phối độ dài.
- Phiên bản: v0 (26/10) = bootstrap + PubMed; v1 (đóng băng 29/10 18:00) = v0 + tự crawl + dự phòng. Sau khi đóng băng, mọi index và embedding đều build trên v1.

## 4. Lưu trữ và truy xuất (D8)

```text
data/raw/official/           query.parquet, links_corpus_vi_en_zh.parquet
data/raw/hf/<dataset>/       bản crawl công khai
data/raw/pubmed/             XML efetch theo lô (gz)
data/raw/html_samples/       HTML mẫu và trang lỗi (gz)
data/interim/crawl_state.sqlite
data/interim/parts/<source>/part-*.parquet
data/processed/corpus/       corpus đã merge
data/processed/coverage.csv
data/processed/docstore.sqlite
```

- Doc store (SQLite `id → title, text, language` hoặc parquet có chỉ mục id) cho pipeline lấy text của vài trăm doc ứng viên mỗi query mà không nạp cả corpus vào RAM.
- Dung lượng ước tính: text ~1,2M doc × ~4 KB ≈ 5 GB dạng thô, ~1,5–2 GB parquet zstd. Embedding doc-level 1024 chiều fp16 ≈ 2,5 GB.
- Toàn bộ `data/` nằm trong `.gitignore`; dùng chung giữa các thành viên qua HF dataset private hoặc Google Drive, không qua git.

## 5. Cấu trúc code đề xuất

```text
src/vibiomir/ingest/
├── __init__.py
├── official.py      tải và đọc query + links chính thức
├── domains.py       domain → ngôn ngữ; cấu hình rate limit theo domain
├── state.py         SQLite crawl state (resume)
├── fetch.py         async fetcher có rate limit và retry
├── extract/         base, trafilatura, sites/<domain>.py, registry
├── pubmed.py        client E-utilities và parser XML
├── bootstrap.py     chuyển dataset HF công khai về schema chung
├── merge.py         hợp nhất nguồn, QA, coverage
└── docstore.py      lấy text theo id
```

- CLI: thêm nhóm lệnh `vibiomir ingest` gồm `download-official`, `manifest`, `pubmed`, `bootstrap --dataset …`, `crawl --domain … --limit …`, `merge`, `coverage`, `export-jsonl --ids …`.
- Optional extra mới `ingest`: `httpx[http2]`, `trafilatura`, `charset-normalizer`, `pyarrow`, `huggingface_hub`. Extra `browser` (`playwright`) chỉ thêm khi thật sự cần.
- Test không gọi mạng: fetcher test bằng transport giả của httpx, extractor test bằng HTML fixture, PubMed parser test bằng XML mẫu nhỏ.

## 6. Thứ tự thực hiện

1. D0, D1 (vài giờ): có manifest và bảng domain.
2. D2 và D3 chạy song song: PubMed hoàn tất, bootstrap tiếng Việt cho ~76% → tạo corpus v0 để R0 nộp bài đầu tiên.
3. D4 + D5 trên phần thiếu, theo thứ tự domain có nhiều link thiếu nhất. Tiếng Trung cần tự crawl nhiều nhất: ~190.000 trang chưa có trong bản `lighth0406` (phần `tmh1008` phủ được thì lấy trước).
4. D7 cho các trang còn lại; D6 merge thành v1 và đóng băng 29/10 18:00; D8 doc store.
