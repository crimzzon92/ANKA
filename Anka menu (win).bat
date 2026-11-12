@echo off
REM Anka's start menu (Windows: double-click). Untested on Windows.
cd /d "%~dp0"
if exist "%USERPROFILE%\anka-env\Scripts\python.exe" (
    "%USERPROFILE%\anka-env\Scripts\python.exe" anka_menu.py
) else (
    py anka_menu.py || python anka_menu.py
)
pause
