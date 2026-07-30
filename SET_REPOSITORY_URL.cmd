
@echo off
setlocal
set /p REPO_URL=Enter the public repository URL (https://github.com/OWNER/CB-FRL-IDS): 
python scripts\set_repository_url.py "%REPO_URL%" || exit /b 1
python scripts\validate_repository.py || exit /b 1
echo Repository URL updated and validation passed.
pause
