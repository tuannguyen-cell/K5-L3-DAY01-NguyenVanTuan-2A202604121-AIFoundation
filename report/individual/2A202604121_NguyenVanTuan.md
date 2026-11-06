# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202604121**
- **Họ và tên: Nguyễn Văn Tuấn**
- **SEED = 2176** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.811      0.896        0.889   
2. BoW + MLP (deep learning)         1.000   0.953      0.976        0.974   
3a. Encoder (MiniLM) + kNN           0.927   0.953      0.940        0.974   
3b. Encoder (MiniLM) + LogReg        0.990   0.934      0.961        0.947   
3c. Encoder (MiniLM) + MLP head      0.980   0.943      0.962        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.014       7680.584       0.066   
2. BoW + MLP (deep learning)         5.466      25233.660       0.225   
3a. Encoder (MiniLM) + kNN         107.485         28.436      55.827   
3b. Encoder (MiniLM) + LogReg      107.883         39.439      29.502   
3c. Encoder (MiniLM) + MLP head    108.467         36.698      29.802   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.107           0.0  
2. BoW + MLP (deep learning)          0.268           0.0  
3a. Encoder (MiniLM) + kNN          115.299           0.0  
3b. Encoder (MiniLM) + LogReg        38.842           0.0  
3c. Encoder (MiniLM) + MLP head      37.925           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

Chọn 3 dự đoán lệch nhiều nhất so với kết quả:

1. **Dự đoán `dl_curve` (Thí nghiệm 2.A — Mạng MLP so với LogReg):**
   - *Dự đoán:* Đoán rằng MLP có 128.129 tham số (gấp 64 lần LogReg) sẽ bị overfit nghiêm trọng trên tập train nhỏ, val loss sẽ tăng mạnh và F1 test sẽ kém hơn hoặc bằng LogReg.
   - *Thực tế:* Val loss ổn định suốt 30 epoch. MLP đạt F1 test vượt trội **0.976** (Recall 0.953, Precision 1.000) so với BoW + LogReg chỉ đạt F1 **0.896** (Recall 0.811).
   - *Nguyên nhân lệch:* 64 nơ-ron lớp ẩn kết hợp hàm phi tuyến ReLU giúp MLP học được các tổ hợp đặc trưng đa từ (ví dụ các cặp từ cùng xuất hiện "call" + "claim" + "now") thay vì cộng dồn trọng số đơn lẻ như LogReg, giúp tăng mạnh khả năng phát hiện spam mà không bị overfit.

2. **Dự đoán `tradeoff` (Đánh đổi độ trễ của kNN):**
   - *Dự đoán:* Đoán rằng mô hình 3a (Encoder + kNN) sẽ có tốc độ xử lý nhanh vì không mất công huấn luyện, chỉ việc tra cứu vector.
   - *Thực tế:* Độ trễ p95 của kNN lên tới **115.299 ms/tin**, vượt xa ngưỡng SLA viễn thông (50 ms) và chậm gấp 3 lần so với Encoder + LogReg (p95 = 38.842 ms).
   - *Nguyên nhân lệch:* kNN không train đồng nghĩa toàn bộ chi phí dồn vào pha inference. Mỗi tin nhắn mới đến đều phải tính khoảng cách Cosine với toàn bộ ~3.600 vector trong tập train rồi sắp xếp lấy top-k, biến kNN thành nút thắt cổ chai lớn khi triển khai thực tế.

3. **Dự đoán `hard_eval` (Khả năng chịu đựng bẫy của BoW):**
   - *Dự đoán:* Đoán rằng BoW chỉ bị lừa ở các từ viết hoa hoặc cố tình thay chữ bằng số (obfuscation).
   - *Thực tế:* BoW còn bị tụt điểm nghiêm trọng ở nhóm `ham_looks_spam` (chặn nhầm tin bạn bè vì có từ "WON", "FREE") và nhóm `natural_spam` (bỏ lọt spam viết giọng thân mật đời thường).
   - *Nguyên nhân lệch:* BoW không hiểu được ngữ cảnh câu và tính ngữ nghĩa; mô hình chỉ đếm từ khoá cứng nên rất dễ bị kẻ xấu ngụy trang hoặc vô tình bắt nhầm tin thường.

## 4. Phân tích 50 case khó

- **Mô hình tụt dốc nhiều nhất:** Mô hình **BoW + LogReg** tụt điểm mạnh nhất, đặc biệt ở 2 nhóm case: `ham_looks_spam` (10 case) và `obfuscation` (6 case).
- **Phân tích 2 case cụ thể:**
  1. **Case 1 — Thuộc nhóm `ham_looks_spam`:**
     - *Nội dung:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: `ham`).
     - *Kết quả:* BoW đoán nhầm thành `spam`.
     - *Cơ chế:* Các từ `won`, `congrats`, `call` trong từ điển BoW đều có trọng số dương rất lớn kéo về phía spam. BoW cộng dồn các giá trị này lại mà không nhận ra các từ ngữ thân mật như `lol`, `beer` hay ngữ cảnh bạn bè đang trêu nhau, dẫn đến chặn nhầm (False Positive).
  2. **Case 2 — Thuộc nhóm `obfuscation` (Né từ khoá):**
     - *Nội dung:* `"C0ngr4tulat10ns! Claim your £1000 c.a.s.h"` (Nhãn thật: `spam`).
     - *Kết quả:* BoW đoán nhầm thành `ham`.
     - *Cơ chế:* Kẻ phát tán spam cố tình chèn số `0` và dấu chấm `c.a.s.h`. Khi qua hàm `tokenize` và đối chiếu từ điển 2.000 từ, các từ này trở thành Out-Of-Vocabulary (OOV) $\rightarrow$ vector trả về toàn số 0. Điểm số dự đoán $z = b < 0 \rightarrow$ mô hình kết luận là `ham`, bỏ lọt tin nhắn độc hại (False Negative). Trong khi đó, Encoder MiniLM vẫn hiểu được ngữ nghĩa câu tặng thưởng tiền bạc.

## 5. Thử thách cá nhân

- **Mã sinh viên:** `2A202604121` $\rightarrow$ `SEED = 2176` $\rightarrow$ Nhận **Đề #5**:
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

- **Cách làm:**
  Thiết kế 4 mẫu tin nhắn thử nghiệm và chạy qua 2 pipeline: BoW + LogReg (Phần 1) và Encoder + LogReg (Phần 3b):
  1. *Spam 1 (Né từ khoá/giọng tự nhiên):* `"Hey mate, your urgent bank transfer is pending, click secure-link.biz/pay to verify now"` (Nhãn: `spam`)
  2. *Spam 2 (Chèn ký tự phân tách):* `"C.0.n.g.r.a.t.s! You got a brand new holiday trip, text back YES to claim it"` (Nhãn: `spam`)
  3. *Ham 1 (Chứa từ 'free'):* `"Are you free this weekend to go play soccer with the team?"` (Nhãn: `ham`)
  4. *Ham 2 (Chứa từ 'win'):* `"We win the debate competition today! Let us celebrate tonight"` (Nhãn: `ham`)

- **Kết quả thực nghiệm:**
  - **BoW + LogReg:** Đúng 1/4 tin. Đoán sai Spam 1 và Spam 2 thành Ham (do từ khoá bị biến dạng không nằm trong từ điển), đoán sai Ham 1 thành Spam (do từ "free" kéo xác suất spam lên cao).
  - **Encoder + LogReg:** Đúng 4/4 tin. Phân loại chính xác toàn bộ 4 trường hợp.

- **Giải thích:**
  - BoW hoàn toàn phụ thuộc vào danh sách từ vựng tĩnh 2.000 từ. Khi gặp từ khóa nhạy cảm trong câu bình thường, nó bị "kích hoạt" sai; khi kẻ lừa đảo thay đổi hình thức viết, nó bị "vô hiệu hóa".
  - Ngược lại, Encoder ánh xạ ngữ nghĩa của cả câu vào không gian vector: cụm "free for soccer" nằm ở toạ độ của sinh hoạt/thể thao, còn "holiday trip claim" dù viết cách chữ vẫn nằm gần toạ độ của các chiêu trò lừa đảo trúng thưởng.

## 6. Phương án cho sản phẩm thật

Với bài toán nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, cao điểm 500–1.000 tin/giây) và yêu cầu khắt khe **p95 < 50 ms/tin**:

1. **Lựa chọn mô hình:** Chọn **2. BoW + MLP (Deep Learning)** làm mô hình chính, hoặc thiết lập **Router 2 tầng (BoW + MLP kết hợp Encoder + LogReg)**:
   - **BoW + MLP:** Đạt F1 test cực cao **0.976** (Recall 0.953, Precision 1.000). Tốc độ xử lý siêu nhanh với thông lượng **25.233 tin/s** và độ trễ p95 chỉ **0.268 ms/tin** — vượt xa yêu cầu SLA 50 ms hơn 180 lần. Chi phí API bằng 0 USD.
   - **Encoder + LogReg (3b):** Có F1 = 0.961 và p95 = 38.842 ms (đạt SLA nhưng tiệm cận ngưỡng 50 ms).

2. **Vì sao loại bỏ LLM API:**
   - *Độ trễ:* LLM gọi qua mạng có p95 từ 500 ms đến 2.000 ms, vi phạm nghiêm trọng chuẩn SLA viễn thông.
   - *Chi phí:* Với 10 triệu SMS/ngày, dùng `gpt-6-luna` tốn khoảng **$105/ngày (~ $3.150/tháng)**. Nếu dùng model lớn như `gpt-6.1-terra` chi phí lên tới **hơn $30.000/tháng** — hoàn toàn lãng phí và không khả thi về mặt kinh tế.

3. **Chấp nhận đánh đổi:**
   Khi dùng BoW + MLP, ta chấp nhận việc mô hình có thể bị qua mặt bởi các thủ thuật cố tình bẻ gãy ký tự (obfuscation) để đổi lấy thông lượng khổng lồ và độ trễ cực thấp. Để khắc phục, ta chỉ cần bổ sung một hàm chuẩn hoá regex đơn giản trước khi vectorize (ví dụ: gộp `c.a.s.h` $\rightarrow$ `cash`) thay vì phải dùng LLM đắt đỏ.

## 7. AI log và điều đã học

### AI log
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 1 | Giải thích công thức tính gradient vector hoá của BCE loss cho Logistic Regression trong numpy | Giải thích đạo hàm $X^T(p - y)/n$ và cách cập nhật trọng số | Kiểm chứng bằng hàm test `_t()` của TODO 1.3 và vẽ biểu đồ loss hội tụ |
| Phần 2 | Hướng dẫn triển khai vòng lặp huấn luyện PyTorch chuẩn với `BCEWithLogitsLoss` | Cung cấp 5 bước chuẩn: `opt.zero_grad()`, forward logit, tính loss, `backward()`, `step()` | Kiểm tra số lượng tham số lý thuyết 128.129 khớp với thực tế |
| Phần 3 | Thuật toán tính Cosine Similarity nhanh cho ma trận không dùng vòng lặp Python | Gợi ý chuẩn hoá vector $L_2$ bằng `norm(axis=1)` rồi nhân ma trận $M \cdot q$ | Tự thêm xử lý chống chia cho 0 để tránh mã lỗi NaN |
| Phần 5 | Gợi ý cách thiết kế 4 tin nhắn thử thách cá nhân theo SEED = 2176 (Đề #5) | Gợi ý các cặp câu đối lập: spam né từ khoá và ham chứa "free"/"win" | Tự viết tin nhắn bằng tiếng Anh, đưa vào pipeline chạy thực nghiệm và phân tích kết quả |

### Điều hiểu ra sau buổi thực hành
1. **Accuracy là con số đánh lừa:** Với dữ liệu mất cân bằng (imbalanced data), chỉ cần đoán bừa vào lớp đa số cũng đạt Accuracy cao; Precision, Recall và F1 mới phản ánh đúng rủi ro chặn nhầm và lọt spam.
2. **Không phải lúc nào to hơn cũng tốt hơn:** Mạng MLP đơn giản trên BoW chỉ mất 0.27 ms để phân loại và đạt F1 0.976, đánh bại hoàn toàn các mô hình nặng nề về mặt hiệu năng phục vụ thực tế (production).
3. **Hiểu bản chất 4 thế hệ:** Từ việc người tự trích xuất đặc trưng (BoW) $\rightarrow$ mạng tự tìm tổ hợp từ (MLP) $\rightarrow$ mượn toạ độ ngữ nghĩa pre-train (Embedding) $\rightarrow$ mô hình sinh văn bản bằng prompt (LLM). Mỗi thế hệ giải quyết một bài toán ngữ nghĩa tốt hơn nhưng luôn đi kèm cái giá phải trả về tài nguyên tính toán và độ trễ.
