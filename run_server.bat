@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%" || exit /b 1

set "LOG_DIR=%ROOT%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\startup.log"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=py -3"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON=python"
        )
    )
)

if not defined VIETOCR_MODEL set "VIETOCR_MODEL=vgg_transformer"
set "VIETOCR_DEVICE=cuda:0"
set "CUDA_VISIBLE_DEVICES=0"
if not defined VIETOCR_BEAMSEARCH set "VIETOCR_BEAMSEARCH=true"

echo ==== %DATE% %TIME% ====>> "%LOG_FILE%"
whoami>> "%LOG_FILE%"
echo Starting VietOCR API from %ROOT%>> "%LOG_FILE%"
echo Python: %PYTHON%>> "%LOG_FILE%"
echo Model: %VIETOCR_MODEL%>> "%LOG_FILE%"
echo Device: %VIETOCR_DEVICE%>> "%LOG_FILE%"
echo CUDA_VISIBLE_DEVICES: %CUDA_VISIBLE_DEVICES%>> "%LOG_FILE%"
echo Beamsearch: %VIETOCR_BEAMSEARCH%>> "%LOG_FILE%"
echo Log file: %LOG_FILE%>> "%LOG_FILE%"

if not defined PYTHON (
    echo No Python interpreter found. Install Python or create .venv first.>> "%LOG_FILE%"
    exit /b 1
)

%PYTHON% -m uvicorn app.main:app --host 0.0.0.0 --port 8085 >> "%LOG_FILE%" 2>&1

endlocal
