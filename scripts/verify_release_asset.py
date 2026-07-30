
from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/verify_release_asset.py PATH_TO_ZIP")
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"Not found: {p}")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
with zipfile.ZipFile(p) as zf:
    bad = zf.testzip()
    names = zf.namelist()
print(json.dumps({"file": str(p), "sha256": h.hexdigest(), "zip_entries": len(names), "bad_entry": bad, "status": "PASS" if bad is None else "FAIL"}, indent=2))
raise SystemExit(0 if bad is None else 1)
