from pathlib import Path
import csv, hashlib, json, sys, yaml
root=Path(__file__).resolve().parents[1]
errors=[]
def req(c,m):
    if not c: errors.append(m)
cfg=yaml.safe_load((root/'config/R3_PROTOCOL_LOCK.yaml').read_text())
req(cfg['status']=='LOCKED_BEFORE_NEW_RESULTS','protocol status')
expected={
 'R3_PRIMARY_CELL_MANIFEST.csv':980,
 'R3_HETEROGENEITY_CELL_MANIFEST.csv':240,
 'R3_NATURAL_PARTITION_CELL_MANIFEST.csv':60,
 'R3_OOD_STRESS_CELL_MANIFEST.csv':60,
 'R3_PRIVACY_CELL_MANIFEST.csv':480,
 'R3_ROBUSTNESS_CELL_MANIFEST.csv':360,
 'R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv':20,
}
for name,n in expected.items():
    rows=list(csv.DictReader(open(root/'manifests'/name,encoding='utf-8-sig')))
    req(len(rows)==n,f'{name}: {len(rows)} != {n}')
    ids=[r['cell_id'] for r in rows]; req(len(ids)==len(set(ids)),f'duplicate ids {name}')
req(cfg['primary_design']['budget_ratios_beta']==[0.1,0.2,0.4,0.6,0.8,1.0],'beta grid')
req(cfg['inference']['f1_noninferiority_margin']==0.02,'F1 margin')
req(cfg['inference']['fpr_safety_margin_absolute']==0.0025,'FPR margin')
req(cfg['novelty']['core']=='excluded','novelty core exclusion')
h=list(csv.DictReader(open(root/'manifests/R3_HYPOTHESIS_MANIFEST.csv',encoding='utf-8-sig'))); req(len(h)==108,'hypothesis count')
req(all(r['status']=='LOCKED' for r in h),'hypothesis lock status')
comp=list(csv.DictReader(open(root/'validation/R2_R3_PYTHON_COMPILE_AUDIT.csv',encoding='utf-8-sig'))); req(all(r['status']=='PASS' for r in comp),'python compile audit')
# Check R2 unit result is embedded/available
r2=json.load(open(root/'validation/R2_TEST_OUTPUT.json'))
req(r2['status']=='PASS','R2 unit tests')
# manifest hashes
mp=root/'PACKAGE_FILE_SHA256.csv'
if mp.exists():
 for r in csv.DictReader(open(mp,encoding='utf-8-sig')):
  p=root/r['relative_path']; req(p.exists(),f'missing {r["relative_path"]}')
  if p.exists():
   h=hashlib.sha256(p.read_bytes()).hexdigest(); req(h==r['sha256'] and p.stat().st_size==int(r['size_bytes']),f'hash mismatch {r["relative_path"]}')
result={'status':'PASS' if not errors else 'FAIL','errors':errors,'counts':expected}
print(json.dumps(result,indent=2)); sys.exit(1 if errors else 0)
