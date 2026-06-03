@echo off
setlocal
cd /d "%~dp0"

title Grid Operations Console

echo.
echo ====================================================
echo   Grid Operations Console - launching...
echo ====================================================
echo.

REM Make sure dependencies are present (silent if already installed)
echo Checking dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] pip install failed. Make sure Python is on PATH.
    echo.
    pause
    exit /b 1
)

REM Train the model the first time (or if it was deleted)
if not exist "models\rf_fault_classifier.pkl" (
    echo Training model for the first time. This takes about 30 seconds.
    python train.py
    if errorlevel 1 (
        echo.
        echo  [ERROR] train.py failed.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Starting dashboard on http://localhost:8501
echo Opening your browser in 5 seconds. Keep this window open.
echo To stop the dashboard, close this window or press Ctrl+C.
echo.

REM Open the browser shortly after Streamlit starts
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"

REM Launch the dashboard (force browser-friendly mode, override config)
python -m streamlit run app.py --server.headless=false --server.port=8501 --browser.gatherUsageStats=false

echo.
echo Dashboard stopped. Press any key to close this window.
pause >nul
