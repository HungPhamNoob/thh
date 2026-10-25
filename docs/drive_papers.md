# Tài liệu tham khảo trong Drive "guru"

Thư mục Drive của Hưng: <https://drive.google.com/drive/folders/1rd9NqjdFfebA5M8cKUBzDOMogBZhSJvJ> (cập nhật 15–16/10/2026). File này là mục lục để agent biết paper nào có sẵn; nội dung đã ánh xạ vào code nằm trong [`source_map.md`](source_map.md).

## Cấu trúc

| Mục | Nội dung |
|---|---|
| `00_TOP30_SOURCES.txt` | Danh sách 26 paper đã qua phản biện + 4 repo GitHub, kèm link venue/PDF. Đây là nguồn của bảng trong `source_map.md`. |
| `PDF_papers_TOP30_peer_reviewed/` | PDF của 26 paper trong danh sách trên. |
| `R2AI2026_papers_2026-10-15/` | 30 PDF arXiv khác, đánh số `01`–`30` (bảng bên dưới). |

## Ghi chú rút ra từ `00_TOP30_SOURCES.txt`

- Nhóm nên dùng trước: 01 BGE-M3, 02 mGTE, 03 ViRE, 04 Vi–En CLIR, 05 MedCPT, 06 BMRetriever, 09 ColBERTv2, 10 PLAID, 12 ColBERT-XM, 18 ConTEB/InSeNT, 23 PARADE.
- Có điều kiện: 16 Query2doc, 17 GRF. Paper gốc dùng LLM đóng; chỉ lấy kỹ thuật, thay bằng LLM mở hợp lệ.
- Model chỉ có tiếng Anh (MedCPT, BMRetriever, SapBERT) chỉ hợp với nhánh PubMed hoặc query đã dịch sang tiếng Anh.
- Kết quả SOTA trong paper đo trên benchmark của paper, không phải điểm R2AI2026. Phải đo lại bằng macro-F2 doc/chunk.
- 4 repo: FlagOpen/FlagEmbedding (BGE-M3), stanford-futuredata/ColBERT (ColBERTv2/PLAID), castorini/pyserini (BM25/Lucene), QwenLM/Qwen3-Embedding (embedding + reranker 0.6B/4B/8B).

## Thư mục `R2AI2026_papers_2026-10-15/`

Cột "Dùng vào việc gì" chỉ điền cho paper đã biết nội dung. Paper đánh dấu "chưa tóm tắt" thì mới chỉ biết tên file; ai cần thì đọc PDF rồi bổ sung vào bảng.

| # | File | Dùng vào việc gì |
|---:|---|---|
| 01 | BGE-M3 (2402.03216) | Dense + learned sparse + multi-vector đa ngữ; đã có adapter `retrieval/bge_m3.py` |
| 02 | Qwen3 Embedding (2506.05176) | Embedding và reranker đa ngữ 0.6B/4B/8B (phát hành 06/2025, hợp lệ); ứng viên chính cho dense và rerank |
| 03 | MedCPT (2307.00589) | Retriever + cross-encoder y sinh tiếng Anh cho nhánh PubMed |
| 04 | MMed-Bench-IR (2606.24200) | Benchmark IR y khoa đa ngữ (chưa tóm tắt) |
| 05 | Multilingual E5 (2402.05672) | Baseline embedding đa ngữ (mE5 large/instruct) |
| 06 | ColBERT-XM (2402.15059) | Late interaction đa ngữ |
| 07 | Jina-ColBERT-v2 (2408.16672) | Late interaction đa ngữ, context dài |
| 08 | ColBERTv2 (2112.01488) | Late interaction + nén vector |
| 09 | PLAID (2205.09707) | Engine tìm kiếm ColBERT trên corpus lớn |
| 10 | PARADE (2008.09093) | Gộp điểm passage thành điểm document; tương ứng `aggregation/` |
| 11 | Late Chunking (2409.04701) | Encode cả tài liệu rồi pool theo chunk, giúp chunk giữ ngữ cảnh |
| 12 | BMQExpander (2508.11784) | Mở rộng query y sinh (chưa tóm tắt) |
| 13 | SapBERT (2010.11784) | Từ đồng nghĩa khái niệm y sinh (UMLS) |
| 14 | NLLB (2207.04672) | Dịch query VI→EN/ZH; tương ứng `query/translation.py` |
| 15 | SPLADE v2 (2109.10086) | Learned sparse retrieval |
| 16 | CeQe (2608.00452) | Query expansion (chưa tóm tắt) |
| 17 | Query2doc (2303.07678) | Sinh pseudo-document để mở rộng query |
| 18 | GPL (2112.07577) | Sinh query giả + pseudo-label để thích nghi domain |
| 19 | InPars-v2 (2301.01820) | Sinh dữ liệu train retriever bằng LLM mở |
| 20 | RAPTOR (2401.18059) | Cây tóm tắt phân cấp; chỉ dùng summary làm key index |
| 21 | LongRAG (2406.15319) | Đơn vị truy hồi dài; liên quan trực tiếp đến việc chọn độ dài chunk |
| 22 | BioASQ14b-BITUA (2609.04999) | Hệ thống tham gia BioASQ 14b (chưa tóm tắt) |
| 23 | BioASQ14b-ARC (2608.01468) | Hệ thống tham gia BioASQ 14b (chưa tóm tắt) |
| 24 | GRF (2304.13157) | Sinh nhiều kiểu feedback (keyword, entity, reasoning) rồi fuse |
| 25 | MIRACL (2210.09984) | Benchmark và cách đánh giá retrieval đa ngữ |
| 26 | Crosslingual-training (2408.10536) | Huấn luyện retriever xuyên ngôn ngữ (chưa tóm tắt) |
| 27 | Dynamic-RAPTOR (2410.01736) | Biến thể RAPTOR (chưa tóm tắt) |
| 28 | Clinical-reranking (2610.01324) | Rerank tài liệu lâm sàng (chưa tóm tắt) |
| 29 | SAFE-QE (2610.13810) | Query expansion (chưa tóm tắt) |
| 30 | InPars-plus (2508.13930) | Sinh dữ liệu train retriever (chưa tóm tắt) |

Trước khi dùng model từ bất kỳ paper nào, kiểm tra ngày phát hành checkpoint (< 01/08/2026) và số tham số (≤ 15B) trên model card, rồi ghi lại tên repo và revision trong config.
