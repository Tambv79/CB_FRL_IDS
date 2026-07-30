# R3 — Protocol Lock

Trạng thái: `PASS_PROTOCOL_LOCKED_READY_FOR_R4_PACKAGE_BUILD`.

Chạy kiểm tra:

```powershell
python .\scripts\validate_r3_protocol.py
```

R3 khóa trước kết quả: datasets, splits, 10 paired seeds, beta grid, baselines, oracle, thresholds, endpoints, statistical families, heterogeneity, privacy và robustness. Không được sửa sau khi test output được truy cập nếu chưa tạo amendment và protocol checksum mới.
