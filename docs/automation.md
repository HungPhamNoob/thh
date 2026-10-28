# Auto pipeline: crawl → index → submission không cần người trực

Tài liệu cho `scripts/auto_pipeline.py` (ticket R0/E2/E4), chạy trên laptop Linux
RTX 5060 8 GB từ 28/10/2026. Nguyên tắc: mọi job dài đều **resume được**, chạy
**tách khỏi terminal** (đóng Claude Code/SSH vẫn chạy tiếp), và **không cần agent
theo dõi realtime** — cron watchdog tự khởi động lại khi daemon chết giữa chừng.

## 1. Chạy

```bash
.venv/bin/python scripts/auto_pipeline.py start     # daemon detached, resume từ stage chưa xong
.venv/bin/python scripts/auto_pipeline.py status    # bảng stage + side job + disk (+ --full: tail log, ingest status)
.venv/bin/python scripts/auto_pipeline.py stop      # dừng daemon; --all dừng cả side job
.venv/bin/python scripts/auto_pipeline.py run       # chạy foreground để debug
.venv/bin/python scripts/auto_pipeline.py watchdog  # cron gọi; chỉ restart khi daemon chết mà chưa DONE/FAILED
```

Cron đã cài cho user `hung` (mỗi 15 phút):

```cron
*/15 * * * * cd /home/hung/YEAR_4/thh && .venv/bin/python scripts/auto_pipeline.py watchdog >> runs/auto/logs/watchdog.log 2>&1
```

Watchdog **không** tự khởi động lại khi: chuỗi đã xong (`runs/auto/DONE`), đã fail
cứng (`runs/auto/FAILED` — cần người/agent xem log rồi quyết định), hoặc quá
`VIBIOMIR_AUTO_DEADLINE` (mặc định 31/10 22:00). Daemon được bọc
`systemd-inhibit --what=idle:sleep:handle-lid-switch` để laptop không ngủ giữa
job; gập máy vẫn crawl bình thường, tắt máy thì chạy lại `start` (mọi stage và
side job đều resume).

## 2. Chuỗi stage

| # | Stage | Nội dung | Ước lượng |
|---|---|---|---|
| 1 | `prepare` | tải `query.parquet` + `links_corpus` (pin revision), viết manifest | vài phút |
| 2 | `pilot` | crawl 5 URL/domain + audit HTML (seed cố định) | ~10 phút |
| 3 | `pilot_gate` | chặn crawl full nếu tỉ lệ ok+short < 30% | tức thì |
| 4 | `crawl` | crawl toàn bộ (trừ pubmed, vov.vn), retry 2 lần | ~8–12 giờ |
| 5 | `pubmed_gate` | chờ side job `pubmed` (tối đa `VIBIOMIR_PUBMED_WAIT_SECONDS`, mặc định 4 h) rồi đếm part; < 95% vẫn merge, EN tạm `missing` | ≤4 giờ |
| 6 | `merge_v1a` | merge crawl+pubmed → `data/processed/corpus/` | ~2 phút |
| 7 | `store_v1a` | `index prepare` → `artifacts/full/store_v1a.sqlite` | ~30–60 phút |
| 8 | `bm25_index_v1a` | export JSONL **gz** + Lucene index, dọn JSONL sau khi build | ~1–3 giờ |
| 9 | `bm25_run_v1a` | `indexed-run` 00_bm25_lucene → `runs/auto/bm25_v1a/submission.json` | ~30 phút |
| 10 | `bm25_package_v1a` | validate substring cả corpus (streaming) + ZIP | ~10 phút |
| 11 | `dense_index_v1a` | encode mGTE → Qdrant Cloud, **GPU ≤ 40%**, retry 2 | ~6–14 giờ |
| 12–13 | `dense_run_v1a`, `dense_package_v1a` | như 9–10 với 01_mgte_dense_auto | ~1 giờ |
| 14+ | nhóm `v1b` | gate: Wayback xong **trước** `VIBIOMIR_V1B_DEADLINE` (mặc định 30/10 18:00) → merge v1b, xoá store v1a, build lại BM25, chạy, đóng gói | ~4–6 giờ |
| cuối | nhóm `densev1b` | chỉ khi bật `VIBIOMIR_DENSE_V1B=1` (tốn ~10 giờ GPU): reindex `--recreate`, chạy, đóng gói | ~10 giờ |

Stage `record_run_info` lưu thêm `run_info.json` (git commit, config, thời điểm) và
`config_resolved.yaml` cạnh mỗi `submission.json` — đúng quy ước `runs/` ở
`AGENTS.md` §6 (E4).

**Side job** (process riêng, pid ở `runs/auto/jobs/`, log `runs/auto/logs/job_*.log`):

- `pubmed`: 240.000 PMID qua E-utilities trong vòng lặp retry (15 lần, ngủ 20 phút
  giữa các lần — NCBI có thể chặn IP tạm thời; batch fail không ghi part nên đếm part
  là độ phủ thật). Thêm `NCBI_API_KEY`/`NCBI_EMAIL` vào `.env` (P0) rồi restart job
  để chạy 10 req/s;
- `vov`: crawl vov.vn tốc độ 0,05 rps theo `Crawl-delay: 20` (~3 ngày, chạy nền tới hạn);
- `wayback`: nguồn dự phòng cho trang chết/bị chặn, khởi động **sau** khi `crawl` xong (~25–40 giờ);
- `slow_retry`: dantri.com.vn + www.qdnd.vn ở 0,5–1 rps.

Daemon restart sẽ **relaunch** side job đã chết giữa chừng (tất cả đều resume-safe),
mỗi job tối đa một lần cho mỗi lượt chạy daemon.

## 3. GPU ≤ 40%

- `VIBIOMIR_GPU_MEM_FRACTION=0.4` (daemon tự set, `src/vibiomir/gpu.py`):
  `torch.cuda.set_per_process_memory_fraction(0.4)` ≈ 3,2 GB / 8 GB VRAM, áp dụng
  cho **mọi** loader model (indexing encoders, retriever dense/BGE/transformers,
  reranker HF/Qwen3/CrossEncoder/BGE-M3). Không có CUDA thì no-op.
- Đây là cap **VRAM**; mức'utilisation không cap cứng được, nên batch size trong
  `configs/full/01_mgte_dense_auto.yaml` đã hạ (encoder 48, reranker 8) để job
  encode dài không ghim 100% util liên tục. Model 0,3B fp16 + batch này chạy
  thoải mái trong 3,2 GB.
- Muốn đổi tỉ lệ: export biến khác trước khi `start`, ví dụ
  `VIBIOMIR_GPU_MEM_FRACTION=0.6 python scripts/auto_pipeline.py start`.

## 4. Disk (chỉ ~30 GB trống)

- Mỗi stage khai `min_free_gb`; thiếu disk thì fail sớm thay vì ghi hỏng
  (`status` hiển thị lý do trong state).
- Export Lucene đổi sang `part-00000.json.gz` (Anserini đọc `.json.gz` trực tiếp):
  ~8 GB → ~2 GB; JSONL bị xoá ngay sau khi index xong.
- `free_disk_v1b` (chỉ trong nhóm v1b): nếu free < `VIBIOMIR_PUBMED_RAW_KEEP_GB`
  (mặc định 16) thì xoá `data/raw/pubmed/` (XML gz chỉ để parse lại khi cần; text
  đã nằm trong parts parquet).
- Qdrant: `01_mgte_dense_auto.yaml` đặt `store_text_payload=false` (payload debug
  chiếm phần lớn dung lượng cloud); submission luôn đọc text nguyên văn từ
  SQLite store nên không ảnh hưởng bất biến chunk_text.

## 5. Đầu ra và việc còn lại cho con người

- `runs/auto/{bm25_v1a,mgte_v1a,bm25_v1b,mgte_v1b}/submission.zip` — ZIP đúng một
  file JSON ở gốc, đã validate schema + substring nguyên văn với cả corpus.
- **Nộp bài vẫn là thao tác thủ công** trên Codabench (không có API nộp): tải
  ZIP lên *My Submissions*, rồi ghi một dòng vào `docs/submission_log.md`.
- Khi chuỗi fail: đọc `runs/auto/FAILED`, `runs/auto/logs/<stage>.log`, sửa, rồi
  `start --from <stage>` (hoặc xoá `runs/auto/state/<stage>.json`).

## 6. Khác biệt với máy Windows cũ

`scripts/start_crawl.ps1`/`stop_crawl.ps1`/`keep_awake.ps1` là của máy Windows
(26–27/10). Trên máy Linux này dùng `auto_pipeline.py`; vai trò tương đương:
Win32_Process.Create → `start_new_session=True`, keep_awake → `systemd-inhibit`,
taskkill → `stop [--all]` (SIGTERM cả process group).
