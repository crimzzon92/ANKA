@echo off
REM Anka's start menu (Windows: double-click). Untested on Windows.
cd /d "%~dp0"
if exist "%USERPROFILE%\anka-env\Scripts\python.exe" (
    "%USERPROFILE%\anka-env\Scripts\python.exe" app\anka_menu.py
) else (
    py app\anka_menu.py || python app\anka_menu.py
)
pause
