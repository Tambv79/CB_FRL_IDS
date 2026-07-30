from pathlib import Path
import json, subprocess, sys, yaml
root=Path(__file__).resolve().parents[1]
cmd=[sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-p','test_*.py','-v']
p=subprocess.run(cmd,capture_output=True,text=True)
result={'status':'PASS' if p.returncode==0 else 'FAIL','returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(root/'validation/R2_TEST_OUTPUT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2)); sys.exit(p.returncode)
