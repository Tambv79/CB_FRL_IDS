
from __future__ import annotations
import json
import sys
from pathlib import Path

OLD = "https://github.com/REPLACE_WITH_GITHUB_USERNAME/CB-FRL-IDS"

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/set_repository_url.py https://github.com/OWNER/CB-FRL-IDS")
new = sys.argv[1].rstrip("/")
if not new.startswith("https://github.com/") or new.count("/") < 4:
    raise SystemExit("Expected a full GitHub repository URL, e.g. https://github.com/OWNER/CB-FRL-IDS")
root = Path(__file__).resolve().parents[1]
changed = []
for p in root.rglob("*"):
    if not p.is_file() or ".git" in p.parts:
        continue
    if p.suffix.lower() not in {".md", ".cff", ".json", ".yaml", ".yml", ".txt"}:
        continue
    text = p.read_text(encoding="utf-8")
    if OLD in text:
        p.write_text(text.replace(OLD, new), encoding="utf-8")
        changed.append(str(p.relative_to(root)))
print(json.dumps({"repository_url": new, "changed_files": changed}, indent=2))
