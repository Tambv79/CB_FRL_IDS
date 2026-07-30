# R2 — Method Lock

Trạng thái: `PASS_METHOD_LOCKED`.

Chạy kiểm tra:

```powershell
python .\scripts\run_r2_validation.py
```

R2 không chứa kết quả mới. Nó khóa utility, eligibility, feasibility, fallback, score, budget semantics và loại novelty khỏi core.
