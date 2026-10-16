# Mapping 26 nguồn → giả thuyết và code

Các PDF trong Drive được dùng để chọn interface và config. Bảng ghi rõ phần đã triển khai trực tiếp và phần cần hạ tầng training/index riêng.

| # | Nguồn, venue | Kỹ thuật lấy vào project | Vị trí |
|---:|---|---|---|
| 1 | BGE-M3, Findings ACL 2024 | một checkpoint cho dense, learned sparse, multi-vector; weighted fusion | `retrieval/bge_m3.py`, config 02 |
| 2 | mGTE, EMNLP Industry 2024 | long-context multilingual encoder và reranker | configs 01, 03 |
| 3 | ViRE, Findings EACL 2026 | giữ BM25 mạnh cho tiếng Việt; đánh giá lexical/dense/hybrid tách biệt | configs 00, 01 |
| 4 | Vi–En CLIR, NAACL 2025 | translate query và multilingual representation | `query/translation.py`, config 04 |
| 5 | MedCPT, Bioinformatics 2023 | asymmetric query/article towers + biomedical cross encoder | config 04, `asymmetric_dense` |
| 6 | BMRetriever, EMNLP 2024 | biomedical instruction-tuned dense retriever | config 09 |
| 7 | SapBERT, NAACL 2021 | entity/synonym expansion từ ontology | `BiomedicalSynonymExpander` |
| 8 | BioLORD, Findings EMNLP 2022 | thêm definition/alias cho concept representation | synonym file + dense ablation |
| 9 | ColBERTv2, NAACL 2022 | token-level MaxSim | BGE-M3 `mode: colbert`, config 06 |
| 10 | PLAID, CIKM 2022 | full-corpus multi-vector cần pruning/index chuyên dụng | production extension note |
| 11 | XTR, NeurIPS 2023 | retrieve token quan trọng trước khi full MaxSim | candidate-token extension note |
| 12 | ColBERT-XM, COLING 2025 | multilingual late interaction | config 06; thay checkpoint multi-vector |
| 13 | GPL, NAACL 2022 | synthetic query + cross-encoder pseudo labels + hard negatives | `docs/training_recipes.md` |
| 14 | SWIM-IR, NAACL 2024 | multilingual synthetic query pairs | `LocalLLMQueryExpander`, training recipe |
| 15 | DR.DECR, NAACL 2022 | translate-retrieve teacher và distillation | config 04, training recipe |
| 16 | Query2doc, EMNLP 2023 | pseudo-document query expansion | `mode: pseudo_document`, config 05 |
| 17 | GRF, SIGIR 2023 | fuse keywords/entities/facts/reasoning views | `LocalLLMQueryExpander`, config 05 |
| 18 | ConTEB/InSeNT, EMNLP 2025 | chunk embedding có document context | `ContextualWindowChunker`, config 03 |
| 19 | RAPTOR, ICLR 2024 | hierarchical/recursive segmentation; summaries chỉ dùng index-time | `RecursiveChunker`; tree summarization recipe |
| 20 | BRIGHT, ICLR 2025 | explicit reasoning reformulation trước retrieval | `mode: reasoning`, config 05 |
| 21 | MIRACL, TACL 2023 | evaluation theo language slice, multilingual retrieval | experiment guide |
| 22 | SPLADE++, SIGIR 2022 | learned sparse expansion + hard-negative distillation | BGE-M3 sparse baseline; Pyserini impact extension |
| 23 | PARADE, TOIS 2024 | aggregate nhiều passage signals về document | `DocumentAggregator` |
| 24 | BEIR, NeurIPS D&B 2021 | robust BM25 baseline, late interaction/rerank comparison | configs 00–09 |
| 25 | MiLQ, EMNLP 2025 | tạo/đánh giá mixed-language query views | NLLB + composite query processor |
| 26 | Revela, ICLR 2026 | self-supervised retriever learning trực tiếp từ raw corpus | training recipe; dùng upstream trainer |

Không nên coi tất cả kỹ thuật là cộng dồn mặc định. Ví dụ contextual chunking, query translation và generative expansion có thể làm tăng nhiễu; chúng cần ablation riêng theo language và loại query.

