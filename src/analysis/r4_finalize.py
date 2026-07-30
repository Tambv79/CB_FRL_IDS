from __future__ import annotations
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
from r4_common import atomic_json,load_yaml,sha256_file

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=load_yaml(Path(args.config));out=Path(cfg['paths']['output_root']);final=out/'finalizer';final.mkdir(parents=True,exist_ok=True)
 gate=out/'validation/R4_VALIDATION_FINAL.json'
 if not gate.exists() or json.loads(gate.read_text()).get('status')!='PASS': raise RuntimeError('Finalizer blocked because final validation is not PASS')
 include=[]
 for pattern in ['analysis/**/*','validation/**/*','R4_STATUS.json','privacy_cells/*/result.json','privacy_cells/*/CELL_COMPLETE.json','cells/*/*/configuration_lock.json','cells/*/*/round_metrics.csv','cells/*/*/selection_records.csv','cells/*/*/validation_threshold_grid.csv','cells/*/*/test_metrics.json','cells/*/*/summary.json','cells/*/*/CELL_COMPLETE.json']:
  include.extend([p for p in out.glob(pattern) if p.is_file()])
 include=sorted(set(include)); manifest=[]
 for p in include: manifest.append({'relative_path':p.relative_to(out).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
 with (final/'R4_COMPLETED_RESULTS_SHA256.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']);w.writeheader();w.writerows(manifest)
 zpath=final/'CB_FRL_IDS_R4_COMPLETED_RESULTS.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for p in include: z.write(p,p.relative_to(out).as_posix())
  z.write(final/'R4_COMPLETED_RESULTS_SHA256.csv','R4_COMPLETED_RESULTS_SHA256.csv')
 atomic_json(final/'R4_FINALIZER_STATUS.json',{'status':'PASS_PACKAGE_CREATED','files':len(include),'zip':str(zpath),'zip_sha256':sha256_file(zpath),'note':'Finalizer packages every completed artifact without retraining. A partial package remains explicitly partial until final validation passes.'});print(zpath)
if __name__=='__main__':main()
