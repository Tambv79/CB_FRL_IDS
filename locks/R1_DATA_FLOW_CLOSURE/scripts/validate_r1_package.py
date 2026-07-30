from pathlib import Path
import pandas as pd, json, hashlib, csv, sys
ROOT=Path(__file__).resolve().parents[1]
errors=[]
def req(x,m):
    if not x: errors.append(m)
flow=pd.read_csv(ROOT/'evidence/R1_CLOSED_DATA_FLOW.csv')
for ds in ['CICIDS2017','CICIoT2023']:
    d=flow[flow.dataset==ds].sort_values('order')
    req(len(d)==8,f'{ds}: expected 8 flow stages')
    raw=int(d.iloc[0].total); unique=int(d.iloc[2].total); post=int(d.iloc[3].total); cohort=int(d.iloc[5].total); final=int(d.iloc[7].total)
    req(raw-int(d.iloc[2].removed_at_stage)==unique,f'{ds}: raw-dedup arithmetic')
    req(unique-int(d.iloc[3].removed_at_stage)==post,f'{ds}: conflict arithmetic')
    req(post-int(d.iloc[5].removed_at_stage)==cohort,f'{ds}: cohort arithmetic')
    req(cohort-int(d.iloc[6].removed_at_stage)==final,f'{ds}: collision arithmetic')
rev=pd.read_csv(ROOT/'validation/R1_REVIEWER_DATA_FLOW_CLOSURE.csv')
req(len(rev)==4 and rev.status.str.startswith('PASS').all(),'reviewer data-flow closure')
g=json.load(open(ROOT/'R1_GATE.json'))
req(g['status']=='PASS_DATA_FLOW_CLOSED_WITH_LEGACY_METADATA_SUPERSEDED','R1 gate status')
print(json.dumps({'status':'PASS' if not errors else 'FAIL','errors':errors},indent=2)); sys.exit(1 if errors else 0)
