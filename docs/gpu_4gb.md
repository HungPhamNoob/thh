# Ma trận tương thích GPU 4 GiB

Mọi model loader gọi `apply_gpu_memory_limit()` trước khi đưa trọng số lên CUDA. Biến
`VIBIOMIR_GPU_MAX_GIB` mặc định là **3,0 GiB** cho PyTorch allocator, thấp hơn hard limit 4 GiB để
chừa CUDA context và fragmentation. `scripts/run_with_gpu_monitor.py` đo VRAM thực tế của toàn bộ
process tree bằng `nvidia-smi`; chỉ run có file đo và hoàn tất mới được ghi `PASS_4GB`.

| Method | Checkpoint | Precision | Batch / max length trong config | CPU fallback | Trạng thái |
|---|---|---|---|---|---|
| `01_mgte_dense` | `Alibaba-NLP/gte-multilingual-base` + reranker chính chủ | FP16 CUDA | index 16/512; rerank 4/512 | encoder tự chuyển CPU nếu batch 1 vẫn OOM | `NOT_TESTED` |
| `02_bge_m3` | `BAAI/bge-m3` | FP16 CUDA | dense/sparse 2/1024; rerank 2/1024 | chạy CPU được khi đặt device ngoài config | `NOT_TESTED` |
| `03_medcpt_en` | MedCPT query/article/cross encoder | FP16 CUDA | index 16/64 query, 512 passage; rerank 8/512 | có thể ép CPU | `NOT_TESTED` |
| `04_bmretriever_en` | `BMRetriever-410M` | FP16 CUDA | 4/512 | có thể ép CPU | `NOT_TESTED` |
| `05_qwen3` | Qwen3 Embedding/Reranker 0.6B | FP16 CUDA | index 1/2048; rerank 1/2048 | có thể ép CPU nhưng rất chậm | `NOT_TESTED` |
| `06_colbertv2_en` | `colbert-ir/colbertv2.0`, PLAID native | theo upstream | do upstream ColBERT quản lý | chưa kiểm định | `NOT_TESTED` |
| `07_splade_v2_en` | SPLADE++ CoCondenser-EnsembleDistil | model mặc định | 4/256 | có thể ép CPU | `NOT_TESTED` |
| `08_late_chunking_en` | `jina-embeddings-v2-small-en` | FP16 CUDA | một document/window 4096, overlap 512 | có thể ép CPU | `NOT_TESTED` |

Các thay đổi phục vụ 4 GiB không đổi tên hoặc thay checkpoint. mGTE reranker bị giới hạn 512 token
thay vì 1024; Qwen3 passage bị giới hạn 2048 thay vì 4096; Late Chunking dùng cửa sổ 4096 thay vì
8192. Các thay đổi này có thể làm giảm chất lượng ở passage rất dài và phải được ghi trong manifest
run. Late Chunking vẫn encode context dài trước rồi pool đúng span chunk; cửa sổ overlap chỉ là cách
giới hạn activation, không biến nó thành contextual chunking.

Khi CUDA OOM, mGTE encoder giảm batch một nửa, giải phóng cache và retry; batch 1 vẫn OOM thì chuyển
CPU nếu `VIBIOMIR_GPU_CPU_FALLBACK=1`. Reranker giảm batch tới 1 rồi fail rõ
`UNSUPPORTED_4GB_VRAM`; không tạo output giả. Qdrant upsert và offset state được ghi sau từng batch,
nên OOM hay mất kết nối không làm mất các point đã commit.

Mỗi production run ghi:

- `artifacts/full/<method>/gpu_peak_index.json` cho index;
- `runs/<method>/gpu_peak_retrieval.json` cho retrieval/rerank;
- `gpu_report.json` cho peak allocator và số OOM PyTorch.
