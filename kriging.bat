@echo off
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" interp-engine 2>nul
if errorlevel 1 (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" interp-engine 2>nul
)
cd /d "%~dp0"
python kriging.py
