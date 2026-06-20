@echo off
cd /d "%~dp0"

REM Search common conda install locations (works on any Windows machine)
for %%d in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\AppData\Local\miniconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "D:\Programs\miniconda3"
    "D:\Programs\anaconda3"
) do (
    if exist "%%~d\condabin\conda.bat" (
        set "CONDA=%%~d\condabin\conda.bat"
        goto :found_conda
    )
)

REM If we get here, conda wasn't found
echo ============================================
echo  Conda not found.
echo  Please install Miniconda first:
echo  https://docs.conda.io/en/latest/miniconda.html
echo  Then run: conda env create -f environment.yml
echo ============================================
pause
exit /b 1

:found_conda
REM Try interp-engine first (from environment.yml), then fafalab2 (legacy)
call "%CONDA%" run -n interp-engine python -m ui_pyside.main_window 2>nul
if %errorlevel% equ 0 goto :eof

call "%CONDA%" run -n fafalab2 python -m ui_pyside.main_window 2>nul
if %errorlevel% equ 0 goto :eof

echo ============================================
echo  Environment not found.
echo  Run: conda env create -f environment.yml
echo  Then activate: conda activate interp-engine
echo ============================================
pause
