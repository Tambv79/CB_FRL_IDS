
@echo off
setlocal
where git >nul 2>nul || (echo Git is not installed or not in PATH.& exit /b 1)
python scripts\validate_repository.py || exit /b 1
set /p REPO_URL=Enter the public repository URL (https://github.com/OWNER/CB-FRL-IDS): 
git init || exit /b 1
git add . || exit /b 1
git commit -m "Initial public release v1.0.0" || exit /b 1
git branch -M main || exit /b 1
git remote remove origin >nul 2>nul
git remote add origin "%REPO_URL%.git" || exit /b 1
git push -u origin main
pause
