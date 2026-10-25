# Đề bài R2AI2026 – ViBioMIR (Shared Task 3)

Tổng hợp ngày 25/10/2026 từ trang thi Codabench, dataset card trên Hugging Face, leaderboard public và thông báo của BTC gửi các đội cùng ngày (luật khớp chunk mới, tắt kết quả chấm chi tiết). Khi BTC đổi luật, sửa file này trước rồi mới sửa code.

- Trang thi: <https://leaderboard.aiguru.com.vn/competitions/15> (API công khai: `/api/competitions/15/`, leaderboard public: `/api/phases/44/get_leaderboard/`)
- Dataset chính thức: <https://huggingface.co/datasets/AIGuruTinix/ViBioMIR>
- Đội: Hưng, Hoàng (lead), Hiệp.

## 1. Nhiệm vụ

Cho 1.200 câu hỏi y khoa tiếng Việt. Với mỗi câu, hệ thống trả về:

- `relevant_docs`: id các tài liệu liên quan, có thể là tài liệu tiếng Việt, Anh hoặc Trung.
- `relevant_chunks`: các đoạn trích nguyên văn từ những tài liệu đó (`doc_id`, `chunk_text`).

Đây là bài toán truy hồi, không phải chatbot. BTC chỉ phát URL; đội tự crawl, làm sạch, chunk và lập chỉ mục.

## 2. Dữ liệu

| File trên HF | Trường | Số dòng |
|---|---|---:|
| `query.parquet` | `id: int`, `query: str` | 1.200 |
| `links_corpus_vi_en_zh.parquet` | `id: int`, `url: str` | 1.220.598 |

Trang Data của cuộc thi gọi file corpus là `links_corpus.parquet`; tên thật trên HF là `links_corpus_vi_en_zh.parquet`. License CC BY-NC 4.0.

Query: trung vị 18 từ, P95 84 từ, dài nhất ~281 từ. Phần lớn là câu hỏi tư vấn của bệnh nhân, kể nhiều triệu chứng trong một câu, ví dụ id 1: "Em bị viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính…".

Corpus theo ngôn ngữ (phân loại theo domain, đo ngày 25/10):

| Ngôn ngữ | Số link | Số domain | Domain lớn nhất |
|---|---:|---:|---|
| vi | 740.598 | 51 | suckhoecongdongonline.vn 154.503 · suckhoedoisong.vn 85.823 · nhathuoclongchau.com.vn 79.598 · thanhnien.vn 72.111 · laodong.vn 31.282 · phunusuckhoe.giadinhonline.vn 27.577 · vinmec.com 26.129 · medlatec.vn 24.762 |
| en | 240.000 | 1 | pubmed.ncbi.nlm.nih.gov, URL dạng `/<PMID>/` |
| zh | 240.000 | 43 | 120ask.com 98.160 · cnkang.com 40.046 · *.39.net 47.261 (ask.39.net 32.772) · familydoctor.com.cn 26.819 · zysjonline.com 6.591 · zhongyibaodian.net 6.061 |

- `id` nằm trong khoảng 583–7.466.161, không liên tục; mỗi domain gần như chiếm một khối id liền nhau. `id` và `url` đều unique.
- Một số domain có tên không giống tiếng Trung nhưng là trang tiếng Trung: `test.pmphai.com`, `www.wujue.com`, `jb39.com`, `www.zydcd.com`, `qihuangzhishu.com`, `www.baidu.com`. Gán ngôn ngữ theo bảng domain, không đoán theo TLD.
- Có URL là trang chủ hoặc trang chuyên mục (ví dụ `https://suckhoecongdongonline.vn/`), không phải bài viết.

## 3. Metric (trang Evaluation)

**Cấp document.** Với mỗi query: `P = |D ∩ G| / |D|`, `R = |D ∩ G| / |G|`.

**Cấp chunk.** Hai phía được chuẩn hoá: Unicode NFKC → giải mã HTML entity → chữ thường → gộp khoảng trắng → bỏ khoảng trắng hai đầu. Với chunk nộp `c` và chunk vàng `g` cùng query, cùng `doc_id`, tính hai tỉ lệ:

```
overlap_gold(c, g) = LCS_tokens(c, g) / |g|
overlap_pred(c, g) = LCS_tokens(c, g) / |c|
```

- Một chunk nộp là **đúng** nếu tồn tại **ít nhất một** chunk vàng mà **cả hai** tỉ lệ đều ≥ 0,4.
- Điều kiện `overlap_pred` được BTC bổ sung qua thông báo ngày 25/10/2026, áp dụng cho mọi lần chấm sau thông báo. Trước đó chỉ xét `overlap_gold`. Lúc kiểm tra lại cùng ngày, trang Evaluation trên Codabench vẫn chỉ ghi điều kiện cũ; coi thông báo là luật hiện hành.
- `P = số chunk nộp đúng / tổng số chunk nộp`.
- `R = số chunk vàng được tìm thấy / tổng số chunk vàng`.

**F2.** `F2 = 5·P·R / (4·P + R)`, tính **cho từng query** rồi lấy trung bình cộng trên mọi query (macro). Leaderboard xếp theo `FINAL_SCORE = (DOCS_F2 + CHUNKS_F2) / 2`; công thức này khớp với các số đang hiển thị.

### Hệ quả cho thiết kế

1. **F2 là trung bình của F2 từng query**, không phải F2 của P/R trung bình. Ví dụ đội đứng đầu ngày 21/10 có P 0,2752 và R 0,6651; F2 tính từ hai số trung bình này là 0,518, còn leaderboard ghi 0,4909.
2. **Precision chunk không ghép 1–1.** Hai chunk nộp cùng khớp một chunk vàng thì cả hai đều được tính đúng. Evaluator trong repo (`src/vibiomir/evaluation/official.py`) đang ghép 1–1, tính F2 từ P/R trung bình và chỉ xét `overlap_gold`, nên **chưa khớp luật**; xem ticket E1 trong [`plan.md`](plan.md).
3. **Độ dài chunk nộp phải xấp xỉ chunk vàng.** Nếu chunk nộp chứa trọn chunk vàng thì LCS = |g|, nên cần |c| ≤ 2,5·|g|. Nếu chunk nộp nằm gọn trong chunk vàng thì LCS = |c|, nên cần |c| ≥ 0,4·|g|. Nói cách khác, kể cả khi trúng đúng chỗ, chunk nộp chỉ được tính khi dài khoảng 0,4–2,5 lần chunk vàng; ngoài khoảng đó chắc chắn trượt. Chunk dài cỡ cả tài liệu không còn được tính. Vì vậy phải ước lượng cỡ chunk vàng (đoạn văn, vài câu hay vài đoạn) và cắt chunk theo đúng cỡ đó; đây là tham số quan trọng nhất ở cấp chunk.
4. **Cách tách token của LCS chưa được công bố.** Repo đang giả định tách theo khoảng trắng sau chuẩn hoá. Nếu BTC cũng làm vậy, với tiếng Trung (gần như không có khoảng trắng) mỗi đoạn viết liền là một token. Khi đó chunk tiếng Trung chỉ nên cắt tại xuống dòng hoặc khoảng trắng, không cắt giữa câu, và text crawl phải giữ đúng ranh giới khoảng trắng của trang gốc. Với điều kiện `overlap_pred`, chunk tiếng Trung gồm nhiều đoạn chỉ khớp khi chunk vàng cũng có số đoạn tương đương.
5. Chỉ id có trong `links_corpus` mới được tính là nhãn dương.

## 4. Định dạng nộp bài

```json
[
  {
    "id": 1,
    "relevant_docs": ["101", "205"],
    "relevant_chunks": [
      {"doc_id": "101", "chunk_text": "Đoạn trích nguyên văn…", "chunk_order": 0}
    ]
  }
]
```

- Phải có đủ 1.200 query. Nếu một cấp không có kết quả thì vẫn ghi danh sách rỗng.
- `id` là số nguyên. Spec yêu cầu `relevant_docs`/`doc_id` là id số nguyên của `links_corpus`, nhưng ví dụ của BTC lại viết dạng chuỗi. Writer hiện tại ghi chuỗi; kiểm tra kiểu nào được chấm đúng ở lần nộp đầu tiên.
- `chunk_text` phải được trích từ tài liệu gốc, không được viết lại hoặc sinh mới. `chunk_order` là tuỳ chọn, số nguyên không âm.
- Nén thành ZIP chứa **đúng một file `.json` ở gốc**, không có thư mục con, rồi tải lên mục *My Submissions*.
- Giới hạn: public 10 bài/ngày; **private tổng cộng 5 bài**. Bài thiếu file hoặc thiếu câu không được chấm và không bị trừ lượt.
- Từ 25/10/2026, file kết quả chấm chỉ còn điểm tổng và trạng thái chạy. BTC đã tắt phần kết quả chi tiết vì nó làm lộ thông tin đáp án, và đang yêu cầu các đội có dấu hiệu khai thác lỗ hổng này giải trình. Không có phản hồi theo từng query: mọi phân tích lỗi phải dựa trên dev set của đội.
- Mỗi đội chỉ có một tài khoản được nộp bài; mỗi người chỉ thuộc một đội.

## 5. Mô hình và dữ liệu ngoài

- Chỉ dùng model công khai trọng số (Hugging Face hoặc tương tự), **≤ 15B tham số**, **phát hành trước 01/08/2026** (giờ VN). Không dùng LLM đóng (GPT-4o, Gemini, …) trong giải pháp.
- Được dùng dữ liệu ngoài nhưng phải trích dẫn và mô tả nguồn đầy đủ để BTC kiểm tra.
- BTC được quyền dùng code và dữ liệu của bài thắng giải.
- Kết quả chỉ được công nhận khi đội nộp working notes paper mô tả đầy đủ phương pháp.

## 6. Mốc thời gian (giờ Việt Nam)

| Mốc | Thời điểm |
|---|---|
| Public test | 01/10 → **31/10 23:59** (phase Codabench kết thúc 2026-10-31T16:59Z) |
| Private test | 01/11 00:00 → **03/11 23:59** (phase kết thúc 2026-11-03T16:59Z) |
| Hạn nộp hệ thống | 04/11 |
| Công bố kết quả | 11/11 |
| Working notes paper | bắt buộc; hạn chưa công bố |

Cả hai phase chấm trên cùng 1.200 query; hệ thống tự chia câu nào thuộc public hay private.

## 7. Leaderboard public (snapshot 25/10/2026, 50 bài nộp)

| Đội | FINAL | Doc F2 | Chunk F2 | Doc P / R | Chunk P / R |
|---|---:|---:|---:|---|---|
| ongthomiu | 0,4871 | 0,4909 | 0,4832 | 0,275 / 0,665 | 0,276 / 0,644 |
| toptire | 0,4274 | 0,4282 | 0,4266 | 0,241 / 0,606 | 0,238 / 0,605 |
| aichickenteam | 0,4083 | 0,4082 | 0,4083 | 0,244 / 0,552 | 0,245 / 0,551 |
| BM25 Baseline (BTC) | 0,3764 | **0,5587** | 0,1941 | **0,881** / 0,515 | 0,340 / 0,176 |

Các điểm này được chấm theo luật chunk cũ (chỉ xét `overlap_gold`); kiểm tra lại sau thông báo 25/10 thì leaderboard chưa được chấm lại. Doc F2 vẫn so sánh được. Chunk F2 của các bài nộp chunk dài sẽ thấp hơn nếu chấm theo luật mới.

Baseline BM25 của BTC có doc F2 cao nhất bảng nhờ precision 0,88. Điều đó gợi ý mỗi query có ít tài liệu đúng và kết quả lexical top-1 thường trúng. Trong khi đó các đội dẫn đầu trả về quá nhiều tài liệu (P ≈ 0,25). Đòn bẩy lớn nhất: giữ precision cấp doc gần mức baseline, đồng thời tăng recall cấp chunk bằng chunk có kích thước sát chunk vàng.

## 8. Câu hỏi cần gửi BTC

Đầu mối liên hệ ghi trên trang thi: nguyetntm@dagoras.io, linhvtt@dagoras.io.

1. Các bài đã nộp trước thông báo 25/10 có được chấm lại theo luật chunk mới không? Trang Evaluation có được cập nhật không?
2. `LCS_tokens` tách token thế nào, đặc biệt với tiếng Trung?
3. `relevant_docs`/`doc_id` nên ghi số nguyên hay chuỗi?
4. Hạn và mẫu trình bày của working notes paper?
5. Có được dùng các bản crawl corpus do đội khác công bố trên Hugging Face (có trích dẫn) không?
