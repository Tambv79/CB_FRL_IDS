$ErrorActionPreference="Stop"
python "$PSScriptRoot\validate_r1_package.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
