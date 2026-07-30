
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PublicRepositoryTests(unittest.TestCase):
    def test_final_validation_pass(self):
        d = json.loads((ROOT / "validation/R4_VALIDATION_FINAL.json").read_text(encoding="utf-8"))
        self.assertEqual(d["status"], "PASS")
        self.assertEqual(d["errors"], [])
    def test_primary_manifest_cardinality(self):
        with (ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as f:
            self.assertEqual(sum(1 for _ in csv.DictReader(f)), 980)
    def test_privacy_manifest_cardinality(self):
        with (ROOT / "manifests/R3_PRIVACY_CELL_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as f:
            self.assertEqual(sum(1 for _ in csv.DictReader(f)), 480)
    def test_raw_data_absent(self):
        forbidden = {".pcap", ".pcapng", ".parquet", ".npy", ".npz"}
        self.assertFalse(any(p.suffix.lower() in forbidden for p in ROOT.rglob("*") if p.is_file()))

if __name__ == "__main__":
    unittest.main()
