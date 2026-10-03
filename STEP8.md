# Phân tích kết quả benchmark

Benchmark hiện được chạy ở chế độ offline xác định, không gọi LLM thật. Kết quả ghi nhận:

| Bộ benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1.402 | 19.599 | 0,00 | 0,20 | 0 | 0 |
| Standard | Advanced | 4.733 | 39.817 | 1,00 | 1,00 | 551 | 0 |
| Long-context stress | Baseline | 233 | 23.162 | 0,00 | 0,20 | 0 | 0 |
| Long-context stress | Advanced | 1.323 | 18.315 | 1,00 | 1,00 | 466 | 16 |

## Nhận xét

- **Advanced recall tốt hơn Baseline:** Advanced ghi các facts ổn định vào `User.md`, vì vậy vẫn trả lời được câu hỏi trong thread mới. Baseline chỉ giữ lịch sử của thread đang chạy nên không có thông tin để trả lời khi chuyển thread.
- **Advanced tốn hơn trong hội thoại thông thường:** ở Standard, Advanced xử lý 39.817 prompt tokens, khoảng gấp đôi Baseline (19.599). Hồ sơ dài hạn được đưa vào ngữ cảnh tạo thêm overhead; khi hội thoại ngắn, lợi ích của compact chưa đủ bù chi phí này. Số `Agent tokens only` cũng cao hơn trong lần chạy này do phản hồi offline của Advanced có thể nhắc lại nhiều facts trong hồ sơ.
- **Compact có lợi thế khi hội thoại dài:** ở stress test, Advanced compact 16 lần và xử lý 18.315 prompt tokens, thấp hơn Baseline 23.162 khoảng 20,9%. Điều này cho thấy compact chủ yếu giảm lượng lịch sử phải mang theo trong prompt; nó không nhất thiết làm giảm token đầu ra của agent.
- **Memory tăng trưởng và rủi ro:** hồ sơ của Advanced tăng 551 bytes ở Standard và 466 bytes ở stress test trong lần chạy này. Dù dung lượng còn nhỏ, trích xuất nhầm, không cập nhật correction, hoặc lưu quá nhiều sở thích có thể khiến memory phình lên, chứa facts sai/cũ và làm tăng prompt overhead.

## Giới hạn của kết quả

`Agent tokens only` và `Prompt tokens processed` hiện là ước lượng heuristic trong chế độ offline, không phải usage do provider báo cáo. `Response quality` cũng là điểm heuristic dựa chủ yếu vào recall và việc câu trả lời không rỗng; đây không phải đánh giá độc lập bằng judge model. Vì vậy, các con số chỉ nên dùng để kiểm tra hành vi và so sánh tương đối trong bài lab, không đại diện cho chi phí hoặc chất lượng production. Muốn kết luận về LLM thật cần hoàn thiện provider, chạy benchmark live và đánh giá response quality bằng judge hoặc review riêng.