@echo off
cd /d "%~dp0"
D:\Programs\miniconda3\condabin\conda.bat run -n interp-engine python -m ui_pyside.main_window
