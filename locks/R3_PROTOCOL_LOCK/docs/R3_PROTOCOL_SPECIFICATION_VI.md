# R3 — Protocol Lock

## Trạng thái

`LOCKED_BEFORE_NEW_RESULTS`. Không có kết quả mới nào được xem khi tạo protocol này.

## Câu hỏi khoa học chính

1. Tại cùng selected-update uplink budget, selector nào bảo toàn F1 và PR-AUC tốt nhất?
2. CB-Score có tốt hơn Utility-Only, Utility/Cost và Oort-Style-Adapted hay không, hay lợi ích phụ thuộc dataset/budget?
3. Trade-off thay đổi thế nào theo beta, cumulative communication và heterogeneity?
4. Privacy leakage và robustness failure có tái lập được dưới protocol được đặc tả đầy đủ hay không?

## Quy mô manifest đã khóa

- Primary budget trade-off: 980 cells.
- Dirichlet sensitivity bổ sung: 240 cells.
- Natural source-group partition: 60 cells.
- CICIDS file-disjoint OOD stress: 60 cells.
- Privacy attacker evaluations: 480 cells.
- Robustness/model-poisoning/metadata-falsification: 360 cells.
- Feature-removal sensitivity: 20 cells.

## Ranh giới tuyên bố

Uplink budget compliance là implementation check. Bằng chứng khoa học chính là detection/communication trade-off ở cùng beta và cùng cumulative bytes. Không score nào được tuyên bố vượt trội trước khi paired results và Holm-corrected inference hoàn thành.

## Không được thay đổi sau lock

Dataset flow, splits, client assignments, beta grid, methods, seeds, endpoints, margins, threshold rules, attacks và statistical families không được thay đổi sau khi test results được truy cập. Mọi amendment phải tạo protocol ID và checksum mới, đồng thời lưu kết quả đã được nhìn thấy.
