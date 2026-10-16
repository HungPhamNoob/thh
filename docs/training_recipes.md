# Training recipes sau baseline

## GPL cho corpus y sinh

1. Sample passage theo language và domain/source.
2. Open LLM sinh 1–3 query mỗi passage.
3. BM25 + multilingual dense lấy hard negatives.
4. Cross encoder teacher score positive và negatives.
5. Fine-tune bi-encoder bằng margin MSE hoặc contrastive loss.
6. So với checkpoint gốc trên dev F2 và breakdown VI/EN/ZH.

Artifacts nên có schema:

```json
{"query":"...","positive_id":"...","negative_ids":["..."],"teacher_scores":[3.2,0.1,-0.4],"language":"vi"}
```

## SWIM-IR / DR.DECR

- Sinh query tiếng Việt từ passage Anh/Trung và query bản địa từ passage cùng ngôn ngữ.
- Teacher mạnh chạy `translate → monolingual retrieve/rerank`.
- Student multilingual nhận query gốc và distill ranking distribution của teacher.
- Mix human/dev pairs với synthetic pairs; giữ source tag để đo ablation.

## RAPTOR an toàn với luật chunk

Summary node chỉ dùng làm retrieval key. Khi một summary được retrieve, map về leaf chunks và nộp `Chunk.text` của leaf. Không nộp summary do model sinh.

## Revela

Revela thay đổi training graph của language model bằng in-batch attention, không phải một retriever adapter nhỏ. Chạy upstream trainer trên raw corpus, export checkpoint, rồi cấu hình checkpoint đó qua `SentenceTransformerRetriever` hoặc adapter encoder tương thích. So sánh cùng chunking/index/cutoff để cô lập tác động của training.

