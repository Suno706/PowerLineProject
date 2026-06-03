@echo off
cd /d "%~dp0"
echo.
echo ============================================
echo  Power Grid Fault Detection - Dashboard
echo ============================================
echo.

if not exist "models\rf_fault_classifier.pkl" (
    echo Training the model first...
    python train.py
    echo.
)

echo Launching dashboard in your browser...
echo If it does not open automatically, go to http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop the dashboard.
echo.
python -m streamlit run app.py
pause
